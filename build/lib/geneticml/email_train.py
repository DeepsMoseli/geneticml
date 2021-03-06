# -*- coding: utf-8 -*-
"""
Created on Thu May  7 20:32:44 2020

@author: Deeps
send training progress vial Gmail
"""

""" what you need:

    -your gmail email address
    -an app password to give this code access to your email (simply go to your Gmail accounts settings and under security find how to get an app password)
"""

#HOW TO USE 
"""  - put this script in the same folder as the base code you will be running
     - in this code, change sender_email, receiver_emai,password(app password from gmail)
     - import this script: "from email_train import Email_pypy"
     - at the part you want to get an update in your code do: "Email_pypy("Message")"
"""


import smtplib, ssl

class Email_pypy:
    def __init__(self,message):
        self.message = message
        self.send_email()
        
    def send_email(self):
    
        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"
        sender_email = "your_email_address@gmail.com"  # Enter your address
        receiver_email = "your_email_address@hawaii.edu"  # Enter receiver address
        subject = "Training Update"
        password = "your_app_password"
        
        contents = {"From":sender_email,"To":receiver_email, "Subject": subject}
        # Create the plain-text message

        messageall = """
        %s
        """%(self.message)
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, messageall)
