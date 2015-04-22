# this code is mostly stolen from http://pymotw.com/2/smtplib/#module-smtplib
# it can send emails through an SMTP server with TLS authentication
# we use this with an recipe on IFTTT.com to get live updates for longer running
# trainings and to transfer results to a personal Dropbox before automatically
# shutting down the EC2 instance. This way we don't have to keep connected to
# the instance all the time.

import smtplib
import email.utils
from email.mime.text import MIMEText

import mailconfig

def send_email(subject, message, file=None):
    # Prompt the user for connection info
    to_email = mailconfig.recipient
    to_name = mailconfig.recipient_name
    from_email = mailconfig.sender
    from_name = mailconfig.sender_name
    servername = mailconfig.host
    username = mailconfig.username
    password = mailconfig.password

    # Create the message
    msg = MIMEText(message)
    msg.set_unixfrom('author')
    msg['To'] = email.utils.formataddr((to_name, to_email))
    msg['From'] = email.utils.formataddr((from_name, from_email))
    msg['Subject'] = subject

    server = smtplib.SMTP(servername)
    try:
        server.set_debuglevel(False)
        server.ehlo()
        server.starttls()
        server.login(username, password)

        server.sendmail(from_email, [to_email], msg.as_string())
    finally:
        server.quit()

if __name__ == "__main__":
    send_email('test', 'hello again!')
