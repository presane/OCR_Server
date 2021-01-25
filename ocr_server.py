import pickle
import socket
import struct
import cv2
import pre_process as prepro
import easyocr
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

HOST = ''
PORT = 5000

reader = easyocr.Reader(['en'], gpu = False)

while True:

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    #prepro = pre_process()

    conn, addr = s.accept()

    data = b'' ### CHANGED
    payload_size = struct.calcsize("L") ### CHANGED

    # Retrieve message size
    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED

    # Retrieve all data based on message size
    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Extract frame
    frame = pickle.loads(frame_data)

    #read image preecha j. 2020-10-26
    pattern_no = 1
    tmp = reader.readtext(frame,text_threshold=0.80)

    img_crop_data = frame[tmp[2][0][0][1]: (tmp[2][0][2][1]) , tmp[2][0][0][0]: tmp[2][0][2][0]]

    rotated = prepro.skew_correct(img_crop_data)
    process_img = prepro.preprocess_img(rotated, name=None)

    if pattern_no == 1:
        #pattern แบบไม่มีขีด
        #custom_config = r'--oem 1 --psm 8 -c tessedit_char_blacklist=IOi\(\)\;!@#$%^&*_+|" "'
        custom_config = r'--oem 1 --psm 6 --user-patterns engPartFBT.user-patterns -c tessedit_char_blacklist=IOi\(\)\;!@#$%^&*_+|" "'
    elif pattern_no == 2:
        #pattern แบบมีขีด
        #custom_config = r'--oem 1 --psm 8 -c tessedit_char_blacklist=IOi\(\)\;!@#$%^&*_+|" "'
        custom_config = r'--oem 1 --psm 6 --user-patterns engPartTest.user-patterns -c tessedit_char_blacklist=IOi\(\)\;!@#$%^&*_+|" "'
    # custom_config = r'--oem 3 --psm 6 --user-patterns ./engPartNo.user-patterns -c tessedit_char_blacklist=\(\)\;!@#$%^&*_+|'
    partNoStr = pytesseract.image_to_string(process_img, config=custom_config,lang='eng')
    #picture
    print(partNoStr)
    cv2.imwrite('img_partNumberFBT.png',process_img)
    
    img_crop_data = frame[tmp[3][0][0][1]: (tmp[3][0][2][1]), tmp[3][0][0][0]: (tmp[3][0][2][0])]

    rotated = prepro.skew_correct(img_crop_data)
    process_img = prepro.preprocess_img(rotated, name=None)

    #kernel = np.ones((5,5),np.uint8)
    #dilation = cv2.dilate(process_img,kernel,iterations = 1)

    custom_config = r'--oem 2 --psm 6 --user-patterns ./engSerial.user-patterns -c tessedit_char_blacklist=iIO\(\)\;!@#$%^&*_+|" "'
    serialStr = pytesseract.image_to_string(process_img , config=custom_config,lang='eng')

    print(serialStr)
    cv2.imwrite('img_serialModuleFBT.png',process_img)
    
    #pack data into str
    st = partNoStr +','+ serialStr
    serialPart = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff\n]', '', st)

    byt = serialPart.encode()
    #c,addr = s.accept()
    print("Got connection from " + str(addr))
    #data=conn.recv(100000) 
    ret_val = conn.send(byt)

    print ("ret_val={}".format(ret_val))
    
    #s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #s.bind((HOST, PORT))
    #s.send(byt)
    print('Socket complete for partNo , Serial')
    #s.listen(10)
    #print('Socket now listening step 2')

    continue
    # Display
    #cv2.imshow('frame', frame)
    #cv2.waitKey(1)
