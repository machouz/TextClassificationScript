import os
import smtplib
import traceback

import telepot

from system_config import EMAIL_PASSWORD, BOT_TOKEN, USERS, EMAIL_USERNAME


def send_email(from_addr, to_addr_list, cc_addr_list, subject, message, login, password,
			   smtpserver='smtp.gmail.com:587'):
	header = 'From: %s\n' % from_addr
	header += 'To: %s\n' % ','.join(to_addr_list)
	header += 'Cc: %s\n' % ','.join(cc_addr_list)
	header += 'Subject: %s\n' % subject
	message = header + message

	server = smtplib.SMTP(smtpserver)
	server.starttls()
	server.login(login, password)
	server.sendmail(from_addr, to_addr_list, message)
	server.quit()


def send_work_done(dataset, cfg=None, error=None, traceback=None):
	import threading
	if not error:
		threading.Thread(target=send_notification, args=(dataset, cfg, "telegram",), kwargs={}).start()
	else:
		threading.Thread(target=send_error_notification, args=(dataset, error, traceback, "telegram"),
						 kwargs={}).start()


def send_notification(dataset, cfg=None, mode="mail"):
	try:
		if mode is 'mail':
			send_email(from_addr='Python Notification', to_addr_list=['yigalyairn@gmail.com', '3danmi@gmail.com'],
					   cc_addr_list=[], subject=dataset + " finished",
					   message=dataset + " has finished extracting features", login=EMAIL_USERNAME,
					   password=EMAIL_PASSWORD)
		elif mode is 'telegram':
			bot = telepot.Bot(BOT_TOKEN)
			for user in USERS:
			    bot.sendMessage(user, "{0}: {1} has finished extracting features".format(
				                dataset, cfg[cfg.rfind(os.sep) + 1:] if cfg is not None else ""))
			
	except Exception as e:
		print("Failed to send notification", "send_notification")
		print("Error: " + str(e), "send_notification")
		print(str(traceback.format_exc()), "send_notification")


def send_error_notification(dataset, error="", trace="", mode="mail"):
	try:
		msg = "Error Occured in {0} run\n\nError: {1}\n\nTraceBack: {2}".format(dataset, error, trace)
		if mode is 'mail':
			send_email(from_addr='Python Notification', to_addr_list=['yigalyairn@gmail.com', '3danmi@gmail.com'],
					   cc_addr_list=[], subject=" Error Occurred", message=msg, login=EMAIL_USERNAME,
					   password=EMAIL_USERNAME)
		elif mode is 'telegram':
			bot = telepot.Bot(BOT_TOKEN)
			for user in USERS:
			    bot.sendMessage(user, msg)
	except Exception as e:
		print("Failed to send notification", "send_error_notification")
		print("Error: " + str(e), "send_error_notification")
		print(str(traceback.format_exc()), "send_error_notification")
