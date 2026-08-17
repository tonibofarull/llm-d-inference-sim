/*
Copyright 2025 The llm-d-inference-sim Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package common

import (
	"context"
	"encoding/binary"
	"fmt"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/vmihailenco/msgpack/v5"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// ParseEndpointPort splits a ZMQ endpoint string (e.g. "tcp://127.0.0.1:5557")
// into the prefix up to and including the last colon, and the trailing port
// number. ok is false if there is no trailing ":<port>".
func ParseEndpointPort(endpoint string) (prefix string, port int, ok bool) {
	lastColon := strings.LastIndex(endpoint, ":")
	if lastColon < 0 {
		return "", 0, false
	}
	port, err := strconv.Atoi(endpoint[lastColon+1:])
	if err != nil {
		return "", 0, false
	}
	return endpoint[:lastColon+1], port, true
}

// OffsetEndpointPort adds the given offset to the port in a ZMQ endpoint
// string (e.g. "tcp://127.0.0.1:5557"). Returns the original
// endpoint unchanged if parsing fails.
func OffsetEndpointPort(endpoint string, offset int) string {
	prefix, port, ok := ParseEndpointPort(endpoint)
	if !ok {
		return endpoint
	}
	return prefix + strconv.Itoa(port+offset)
}

// EncodeSeq encodes a sequence number as an 8-byte big-endian slice, the
// wire format used for the seq frame in both the live PUB stream and KV
// events replay.
func EncodeSeq(seq uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, seq)
	return b
}

// Publisher sends events to a ZMQ endpoint.
//
// The bind/dial decision mirrors vLLM's ZmqEventPublisher._socket_setup:
// https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/distributed/kv_events.py#L385
// Bind when the endpoint is "stable" (a wildcard or local transport), dial
// otherwise
//   - "tcp://*:5557"    -> bind (server)
//   - "tcp://[::]:5557" -> bind
//   - "ipc:///tmp/x"    -> bind
//   - "inproc://x"      -> bind
//   - "tcp://host:5557" -> dial (client)
type Publisher struct {
	socket   zmq4.Socket
	endpoint string
	seqNum   uint64
}

// NewPublisher creates a new ZMQ publisher.
func NewPublisher(ctx context.Context, endpoint string) (*Publisher, error) {
	p := &Publisher{endpoint: endpoint}
	if err := p.socketSetup(ctx); err != nil {
		return nil, err
	}
	return p, nil
}

func (p *Publisher) socketSetup(ctx context.Context) error {
	p.socket = zmq4.NewPub(ctx,
		// -1 means try forever
		zmq4.WithDialerMaxRetries(-1),
		// reconnect if server restarts
		zmq4.WithAutomaticReconnect(true),
		// wait 1s between attempts
		zmq4.WithDialerRetry(time.Second),
	)

	if shouldBind(p.endpoint) {
		log.FromContext(ctx).Info("ZMQ publisher binding", "endpoint", p.endpoint)
		if err := p.socket.Listen(p.endpoint); err != nil {
			return fmt.Errorf("failed to bind ZMQ publisher: %w", err)
		}
		return nil
	}

	// Connect (dial). zmq4.Dial blocks until connected, so run it in the
	// background: the socket queues sends until the connection comes up and
	// auto-reconnects if the peer restarts.
	go func() {
		log.FromContext(ctx).Info("ZMQ publisher dialing", "endpoint", p.endpoint)
		if err := p.socket.Dial(p.endpoint); err != nil {
			// Context cancellation during shutdown is expected — don't treat it as an error.
			if ctx.Err() != nil {
				return
			}
			log.FromContext(ctx).Error(err, "ZMQ dialer exited", "endpoint", p.endpoint)
			return
		}
		log.FromContext(ctx).Info("ZMQ dialer connected", "endpoint", p.endpoint)
	}()
	return nil
}

// shouldBind reports whether endpoint is "stable" and should be bound rather
// than dialed. Mirrors vLLM's _socket_setup heuristic.
func shouldBind(endpoint string) bool {
	return strings.Contains(endpoint, "*") ||
		strings.Contains(endpoint, "::") ||
		strings.HasPrefix(endpoint, "ipc://") ||
		strings.HasPrefix(endpoint, "inproc://")
}

// PublishEvent marshals batch, assigns the next sequence number, and sends
// [topic, seq, payload] over ZMQ. Returns the assigned sequence number and
// the marshaled payload so callers can store it without re-encoding.
func (p *Publisher) PublishEvent(ctx context.Context, topic string, batch interface{}) (uint64, []byte, error) {
	logger := klog.FromContext(ctx).V(0)

	payload, err := msgpack.Marshal(batch)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to marshal event batch: %w", err)
	}

	// sequence number for ordering
	seq := atomic.AddUint64(&p.seqNum, 1)

	// send topic, sequence, payload
	msg := zmq4.NewMsgFrom([]byte(topic), EncodeSeq(seq), payload)

	if err = p.socket.Send(msg); err != nil {
		return 0, nil, fmt.Errorf("failed to send message to topic %s: %w", topic, err)
	}

	logger.V(logging.TRACE).Info("Published event batch", "topic", topic, "seq", seq)
	return seq, payload, nil
}

// Close closes the publisher and cleans up resources.
func (p *Publisher) Close() error {
	if p.socket != nil {
		return p.socket.Close()
	}
	return nil
}
