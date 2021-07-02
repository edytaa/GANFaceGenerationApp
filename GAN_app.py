import streamlit as st
import sys
import os
import zmq
import time
from random import randint
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from dummy import *

basePth = r'/home/edytak/Documents/GAN_project/code/'
path_stylegan = basePth + r'stylegan2'
sys.path.append(path_stylegan)

imPth = basePth + 'stimuli/'  # path to dir with participants' folders
#imPth = r'/analyse/Project0257/'
init_run = False  # indicates when a session is run for the first time
initial_user_id = 0

NON_FUNCTIONAL_GRADE = 99

def main():
    global init_run
    st.title("How good are you at reconstructing famous faces?")

    expander = st.sidebar.beta_expander("Enter your participant ID")
    user_id = expander.text_input("Participant ID", initial_user_id)
    st.sidebar.write(f'Participant ID: {user_id}')

    message = "Reset: {reset} Rate: {rate} UserID: {userID}"  # template for message for communication with server
    userPth = imPth + user_id + '/'
    socket = load_socket()

    # button to start a new session (for testing)
    new_session_button = st.sidebar.button('New session')

    grades_button_values = []
    grade_range = 10
    for button_val in range(grade_range):
        grades_button_values.append(st.sidebar.button(str(button_val)))
    #option = [i for i in range(grade_range) if grades_button_values[i]][0]
    option = [idx for idx, state in enumerate(grades_button_values) if state]
    option = option[0] if len(option) else NON_FUNCTIONAL_GRADE

    if int(user_id) == 0:
        st.image(imPth + r'images/start.png', width=550)
        message_sent = message.format(reset='False', rate=option, userID=user_id)  # send session state and rating
        socket.send(bytes(message_sent, 'utf-8'))
        received = socket.recv_multipart()
        samples_generated = received[2].decode("utf-8")
        trial = int.from_bytes(received[0], "little")  # bytes to int

    elif new_session_button:
        message_ = message.format(reset='True', rate='None', userID=user_id)
        print(message_)  # print sent message in terminal
        socket.send(bytes(message_, 'utf-8'))

        received = socket.recv_multipart()  # get trial number and generation
        trial = int.from_bytes(received[0], "little")  # bytes to int
        generation = int.from_bytes(received[1], "little")
        samples_generated = received[2].decode("utf-8")
        st.write(f'Generation: {generation}, Trial: {trial}')
        st.image(userPth + 'trl_0_' + str(trial) + '.png', width=550)  # display image

    else:
        message_sent = message.format(reset='False', rate=option, userID=user_id)  # send session state and rating
        socket.send(bytes(message_sent, 'utf-8'))
        received = socket.recv_multipart()  # get trial number and generation
        trial = int.from_bytes(received[0], "little")  # bytes to int
        generation = int.from_bytes(received[1], "little")
        samples_generated = received[2].decode("utf-8")
        n_generated_samples = int.from_bytes(received[3], "little")  # number of rendered samples (part of threading)
        picture_path = userPth + 'trl_' + str(generation) + '_' + str(trial) + '.png'
        if samples_generated == 'False':  # if new samples are not ready yet
            st.write(f'Please wait. New samples are being generated: {trial} / 55')
        elif os.path.isfile(picture_path):
            st.write(f'Previous image rated with: {option}')
            st.write(f'Generation: {generation}, Trial: {trial}')
            st.image(userPth + 'trl_' + str(generation) + '_' + str(trial) + '.png', width=550)  # display image
 #       else:
     #       st.write(f'Hello {user_id}, Click Start to begin experiment')

    print(f'\n samples ready?: {samples_generated}, trial {trial}')


# initialise communication (run only once)
@st.cache(allow_output_mutation=True)
def load_socket():
    global init_run
    init_run = True
    #st.image(imPth + r'images/start.png', width=550)
    print("Connecting to a server…")
    #  Socket to talk to server
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    return socket

if __name__ == "__main__":
    main()
