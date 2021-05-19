import sys
import os
import zmq


def load_socket():
    print("Connecting to a server…")
    #  Socket to talk to server
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    return socket


socket = load_socket()
message = "Reset: {reset} Rate: {rate}"
message_ = message.format(reset='True', rate='None')
print(message_)  # print sent message in terminal
socket.send(bytes(message_, 'utf-8'))
received = socket.recv_multipart()  # get trial number and generation
print(f'received from server: {received}')

for i in range(20):
    message_sent = message.format(reset='False', rate=i%10)  # send session state and rating
    socket.send(bytes(message_sent, 'utf-8'))
    print(f'Previous image rated with: {i%10}')
    received = socket.recv_multipart()  # get trial number and generation
    print(f'received from server: {received}')

print("end of requests")