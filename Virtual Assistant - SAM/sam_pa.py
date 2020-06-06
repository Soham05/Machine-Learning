# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:57:49 2020

@author: shmbh
"""

import speech_recognition as sr
import os
import sys
import re
import webbrowser
import smtplib
import requests
import subprocess
from pyowm import OWM
#import youtube_dl
#import vlc
import urllib.request
import urllib.parse
#import urllib3
#import json

from bs4 import BeautifulSoup as soup
#import wikipedia
import random
# from time import strftime
from gtts import gTTS
from pygame import mixer
import time
import playsound


def samresponse(audio):### convert audio to speech
    
    
    print(audio)
    #x = ['sunny', 'sagar', 'akhil']
    #tts = 'tts'
    for line in audio.splitlines():
        tts = gTTS(text= audio, lang = 'en')
        file1 = str("audio.mp3")
        tts.save(file1)
        playsound.playsound(file1,True)
        #print('after')
        os.remove(file1)
    
    #for i in range(0,3):
      #      text_to_speech = gTTS(text = audio,lang = 'en-us')
       #     file = str(str(i) + "audio.mp3")
        #    text_to_speech.save(file)
         #   mixer.init()
          #  mixer.music.load(file)
           # mixer.music.play()
            #while mixer.music.get_busy() == True:
             #   pass
            #os.remove(file)
            #print('after')
            
        #os.system("say "+ audio)


def myCommand():  ## interpret user voice response
    
    
    
    r = sr.Recognizer()
    
    
    with sr.Microphone() as source:
        print("Say something...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source,duration=1)
        audio = r.listen(source)
        print('analyzing')
    try:
        command = r.recognize_google(audio).lower()
        print("You said: " +command+"\n")
        time.sleep(2)
        #Loops back to continue to listen for commands if unrecognizable commands are recieved
    except sr.UnknownValueError:
        print("Your last command was not recognized")
        command = myCommand()
    return command




def SAM(command):
    errors = ['Ummmm..I am not quite sure what you meant there! Please try again',
              'Sorry, I cannot help you with that!',
              'Come again!',
              'I knew this would happen! I am still learning so please bare with me!',
              'Interesting, I dont have a answer for that. I will definitely  answer that in my next version!But for that you need to fund my master!',
              ]
    
    whoami = ['You dont remember who I am? This must be starting signs of old age',
              'Bond.........James Bond! Just kidding, you cant afford him!',
              'Call me SAM']
    
    if "open" in command:
        reg_ex = re.search('open (.+)',command)
        if reg_ex:
            domain = reg_ex.group(1)
            print(domain)
            url = 'https://www.' +domain
            webbrowser.open(url)
            samresponse('Requested website is openned in the browser')
        
    

            
    elif "current weather in" in command:
        reg_ex = re.search("current weather in (.+)",command)  
        if reg_ex:
            city = reg_ex.group(1)
            url = 'http://api.openweathermap.org/data/2.5/weather?q={}&appid=606335e2c1065985b4e4038dc4008e89&units=metric'.format(city)
            response = requests.get(url)
            data = response.json()
        #print(data)
            temp = data['main']['temp']
            round_temp = int(round(temp))
            samresponse('It is {} degree celcius in {}'.format(round_temp, city))
            time.sleep(3)
        else:
            samresponse('Please say that again')
            
    elif "time" in command: 
        import datetime
        now = datetime.datetime.now()
        samresponse('Current time is %d hours %d minutes'  %(now.hour,now.minute))
    
    elif "shut down" in command:
        samresponse("It was nice working for you! See you next time. Bye!")
        sys.exit()
    
    elif "tell me about" in command:
        reg_ex = re.search("tell me about (.+)",command)
        if reg_ex:
            topic = reg_ex.group(1)
            response = requests.get("https://en.wikipedia.org/wiki/" + topic)
            
            if response is not None:
                html = soup(response.text,'html.parser')
                title = html.select("#firstHeading")[0].text
                paragraph = html.select('p')
                for para in paragraph:
                    print(para.text)
                
                intro = '\n'.join([para.text for para in paragraph[0:5]])
                print(intro)
                
                speech = 'speech.mp3'
                language = 'en'
                myobj = gTTS(text=intro,lang=language,slow='False')
                myobj.save(speech)
                mixer.init()
                mixer.music.load(speech)
                mixer.music.play()
        elif 'stop' in command:
            mixer.music.stop()
    
    
    elif "youtube" in command:
        samresponse('OK!')
        reg_ex = re.search('play (.+) on youtube', command)
        if reg_ex:
            domain = reg_ex.group(1)
            query_string = urllib.parse.urlencode({"search_query":domain})
            html_content = urllib.request.urlopen("https://www.youtube.com/result?"+query_string)
            search_results = re.findall(r'href=\"\/watch\?v=(.{11})', html_content.read().decode()) # finds all links in search result
            webbrowser.open("http://www.youtube.com/watch?v={}".format(search_results[0]))
            pass
    
    #    elif "email" or "gmail" in command:
#        samresponse("What is the subject?")
#        time.sleep(3)
#        subject = myCommand()
#        samresponse("What should I say?")
#        message = myCommand()
#        content = "Subject: {}\n\n{}".format(subject, message)
#    
#    # init gmail SMTP
#        mail = smtplib.SMTP("smtp.gmail.com", 587)
#    
#    # identify to server
#        mail.ehlo()
#    
#            # encrypt session
#        mail.starttls()
#    
#            # login
#        mail.login("shm.bhalerao@gmail.com", "Soham4102015")
#    
#            # send message
#        mail.sendmail("FROM", "TO", content)
#    
#            # end mail connection
#        mail.close()
#    
#        samresponse("Email sent.")
            
        
#    elif "launch" in command:
#        reg_ex = re.search('launch(.+)',command)
#        if reg_ex:
#            appname = reg_ex.group(1)
#            app = appname +'.exe'
#            subprocess.Popen(["open", "-n", "/Applications/" + app], stdout=subprocess.PIPE)
    elif 'hey sam' in command:
        samresponse('hey there. how can i help you?')
        time.sleep(3)
    
    elif 'who are you' in command:
        reply = random.choice(whoami)
        samresponse(reply)
        time.sleep(3)
    
    else:
        error = random.choice(errors)
        samresponse(error)
        time.sleep(3)
        

    
samresponse('SAM activated!')
    
    
while True: ## loop to continue execting multiple commands
    time.sleep(4)
    SAM(myCommand())

            
        
        
    
    
    
    