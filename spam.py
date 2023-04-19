# import pywhatkit
# import datetime
# import itertools

# # set the time at which you want to send the message
# hour = 0
# minute = 10

# # get the current time
# now = datetime.datetime.now()

# # calculate the time delta until the desired time
# delta = datetime.timedelta(hours=hour, minutes=minute) - datetime.timedelta(hours=now.hour, minutes=now.minute)

# # calculate the seconds until the desired time
# seconds = delta.seconds + 1

# # define the message to be sent
# message = "Good morning, students! This is your daily reminder to review today's lesson."

# # send the message to the specified WhatsApp group chat
# for i in itertools.count():

#     pywhatkit.sendwhatmsg_instantly("+2348083337013", message)


import time 
import pywhatkit
import pyautogui
from pynput.keyboard import Key, Controller

keyboard = Controller()


def send_whatsapp_message(msg: str):
    try:
      while True:    
        pywhatkit.sendwhatmsg_instantly(
            phone_no="+2348083337013", 
            message=msg,
            tab_close=False
        )
       ## time.sleep(1)
        pyautogui.click()
      ##  time.sleep(2)
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
        print("Message sent!")
    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    send_whatsapp_message(msg="Test message from a Python script!")