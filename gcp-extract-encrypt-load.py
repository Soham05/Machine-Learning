# -*- coding: utf-8 -*-

import pyodbc
import hashlib, sys, glob,os, md5
from base64 import b64encode, b64decode
from datetime import datetime
from google.cloud import storage
from google.oauth2 import service_account
import subprocess
import json  
import argparse
from time import sleep


current_date=datetime.today().strftime('%Y%m%d')
def upload_file_to_bucket(bucket_name,source_file_name,destination_blob_name):
   client = storage.Client(project='tmo-dms-dev')
   mybucket = client.get_bucket(bucket_name)
   blob = mybucket.blob(destination_blob_name)
   blob.upload_from_filename(source_file_name)
   print("File {} uploaded to {}/{}".format(source_file_name,bucket_name,destination_blob_name))

## Get SQL name as argument

def get_arguments():
    parser = argparse.ArgumentParser(description='get sql file name:')
    parser.add_argument('-sql',help='name of sql script to extract data from',type=str, 
                        dest='-sqlPath',default='.',required=True)
    return parser.parse_args()

def data_extraction(c,sqlName,scripts_path,file_path):
    

    query_file_path=glob.glob(scripts_path +sqlName+'.sql')
    
    
    for file in query_file_path:
        query_file=open(file,'r')
        query=""
        encrypt_list=['FIRST_NAME','LAST_NAME']
        encrypt_index=[]
        encrypted_column=[]
        
        for line in query_file.readlines():
            query+=line[:-1]+" "
        print(query)
        
        
        cur=c.cursor()
        result=cur.execute(query)
        print("\nWriting to file started")
        # New changes added for GCP file uplad automation
        filename_1 = sqlName+'_'+current_date
        filename=filename_1+'.csv'
        #audit_file=filename_1+'.audit'
        data_file_name = file_path+filename
        f=open(data_file_name,'w')
   
    
        column_names=cur.description
        column_names_modified=[]
        cnt=0
        line=''
        for column in column_names:
            cnt+=1
            column_name=str(column[0])
            if (column_name in encrypt_list):
                encrypt_index.append(cnt)
                encrypted_column.append(column_name)
            if column_name in column_names_modified:
                column_name+='_MASTER'
            column_names_modified.append(column_name)
            line+=str(column_name)+"|"
        f.write(line[:-1])
        f.write("\n")
        print(cnt)
        load=0
        while True:
            rows = cur.fetchmany(50000)
            load += 1
            if not rows:
                break
            print("Starting Load No:{}".format(load))
            for i in rows:
                line=''
                cnt2=0
                for j in i:
                    cnt2+=1
                    if str(j)=='None':
                        line+='|'
                    else:
                        if(cnt2 in encrypt_index):
                            #sha256 code
                            hashedvalue=hashlib.sha256(str(j).encode())
                            line+=b64encode(bytes.fromhex(hashedvalue.hexdigest())).decode()+"|"
                        #line+=hashedvalue.hexdigest()+"|"
                        else:
                            line+=str(j)+"|"
                f.write(line[:-1]+"\n")
        	#f.write(str(i)[1:-1].replace('None','').replace(' \'','').replace('\'','')+"\n")
        f.close()
        print("Completed {} Loads".format(load))
        cur.close()
        #########md5 hashvalue for a file #############
        m = md5()
        with open(data_file_name,'rb') as f_in:
            for chunk in f_in:
                m.update(chunk)
        file_md5hashvalue = m.hexdigest()
        print(file_md5hashvalue)
        destination_blob_name = '/inbound/'+filename
        upload_file_to_bucket(bucket_name='dms-scopsbi-data', source_file_name=data_file_name, destination_blob_name=destination_blob_name)

def main():
    print('Get sql file name:\n')
    sqlName = get_arguments()
    print('Extracting data for:', sqlName)
    
    home_path = '/root/'
    scripts_path = home_path + '/scripts/'
    file_path = home_path + '/files/'
    config_path = home_path + '/config/'
    
    # get credentials from config file

    with open(config_path + 'config.json') as conf_file:
        conf = json.load(conf_file)
    
    c = pyodbc.connect('Driver={ODBC Driver 13 for SQL Server};'
                      'Server=*****;'
                      'Database=******;'
                      'UID=********;'
                      'PWD=*******;'
                    )
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']= '*******.json'
    print(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    
    data_extraction(c,sqlName,scripts_path,file_path)
    sleep(400)
    args = ('rm','-rf','/files/*.csv')
    subprocess.call('%s %s %s' % args,shell=True)

    
