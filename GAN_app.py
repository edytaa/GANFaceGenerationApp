import streamlit as st
import sys
import os
import zmq
os.environ["CUDA_VISIBLE_DEVICES"]="0"

basePth = r'/home/edytak/Documents/GAN_project/code/'
path_stylegan = basePth + r'stylegan2'
sys.path.append(path_stylegan)

imPth = basePth + 'stimuli/images/'  # path to generated images
init_run = False  # indicate when a session is run for the first time

def main():
    st.title("GAN App")
    message = "Reset: {reset} Rate: {rate}"  # template for message for communication with server

    socket = load_socket()

    # button to start a new session (for testing)
    new_session_button = st.sidebar.button('New session')

    grades_button_values = []
    grade_range = 10
    for button_val in range(grade_range):
        grades_button_values.append(st.sidebar.button(str(button_val)))
    #option = [i for i in range(grade_range) if grades_button_values[i]][0]
    option = [idx for idx, state in enumerate(grades_button_values) if state]

    option = option[0] if len(option) else 0

    # reset a session
    if new_session_button or init_run:
        message_ = message.format(reset='True', rate='None')
        print(message_)  # print sent message in terminal
        socket.send(bytes(message_, 'utf-8'))

        received = socket.recv_multipart()  # get trial number and generation
        trial = int.from_bytes(received[0], "little")  # bytes to int
        generation = int.from_bytes(received[1], "little")
        st.write(f'Generation: {generation}, Trial: {trial}')
        st.image(imPth + 'trl_0_' + str(trial) + '.png', width=550)  # display image

    elif not init_run:
        message_sent = message.format(reset='False', rate=option)  # send session state and rating
        socket.send(bytes(message_sent, 'utf-8'))
        st.write(f'Previous image rated with: {option}')
        received = socket.recv_multipart()  # get trial number and generation
        trial = int.from_bytes(received[0], "little")  # bytes to int
        generation = int.from_bytes(received[1], "little")
        st.write(f'Generation: {generation}, Trial: {trial}')
        st.image(imPth + 'trl_' + str(generation) + '_' + str(trial) + '.png', width=550)  # display image

    else:
        st.image(imPth + 'trl_0_0.png', width=550)  # display initial image

# initialise communication (run only once)
@st.cache(allow_output_mutation=True)
def load_socket():
    global init_run
    init_run = True
    print("Connecting to a server…")
    #  Socket to talk to server
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    return socket

if __name__ == "__main__":
    main()
