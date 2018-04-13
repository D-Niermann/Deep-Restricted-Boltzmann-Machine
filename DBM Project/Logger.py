# -*- coding: utf-8 -*-
""" 
	Dario Niermann
	    2017

"""


class Logger(object):
	def __init__(self,on):
		self.count        = 1
		self.count_save   = []
		self.start_time   = [0]
		self.end_time     = 0
		self.process_name = [""]
		self.on           = on #true false
		self.write_file   = False #true #false

	def open(self,path):
		if path[-1]!="/":
			path=path+"/"
		if self.on:
			self.file = open(path+"Logger-File.txt","w")
			self.write_file = True

	def info(self,*args):
		if self.on:			
			string="Info     "+"\t"*(len(self.start_time)-1)
			for i in args:
				string+=str(i)+" "
			printstr = string 
			print printstr
			if self.write_file:
				self.file.write(printstr+"\n")


	def out(self,input_str,*args):
		if self.on:
			if self.count==1:
				printstr = ""
				print printstr
				if self.write_file:
					self.file.write(printstr+"\n")
			for i in args:
				input_str+=" "+str(i)
			try:
				printstr = str(self.count)+".      "+"\t"*(len(self.start_time)-1)+str(input_str)
				print printstr
				if self.write_file:
					self.file.write(printstr+"\n")
				self.count+=1
			except:
				printstr = "Error while logging"
				print printstr
				if self.write_file:
					self.file.write(printstr+"\n")
		
	def start(self,input_str,*args):
		if self.on:
			if self.count==1:
				printstr = ""
				print printstr
				if self.write_file:
					self.file.write(printstr+"\n")
			for i in args:
				input_str+=" "+str(i)
			from time import time
			try:
				start_len=(len(self.start_time))
				printstr = str(self.count)+"."+"-"*start_len+"v"+" "*(5-start_len)+"\t"*(len(self.start_time)-1)+str(input_str)
				print printstr
				if self.write_file:
					self.file.write(printstr+"\n")
				self.count_save.append(self.count)
				self.count+=1
				self.start_time.append(time())
				self.process_name.append(input_str)
			except:
				printstr = "Error: while begin logging"
				print printstr
				if self.write_file:
					self.file.write(printstr+"\n")

	def end(self):
		if self.on:
			if self.start_time[len(self.start_time)-1]!=0:
				from time import time
				self.end_time=time()
				try:
					start_len=(len(self.start_time)-1)
					printstr = str(self.count_save.pop())+"."+"-"*start_len+"^"+" "*(5-start_len)+"\t"*(len(self.start_time)-2)+self.process_name[len(self.process_name)-1]+" (took "+str(round(self.end_time-self.start_time[len(self.start_time)-1],3))+" sek)"
					print printstr
					if self.write_file:
						self.file.write(printstr+"\n")
					time_needed=round(self.end_time-self.start_time[len(self.start_time)-1],6)
					self.start_time.pop()
					self.process_name.pop()
					self.end_time=0
					return time_needed
				except:
					printstr = "Error: while end logging"
					print printstr
					if self.write_file:
						self.file.write(printstr+"\n")
			else:
				printstr = "Error : end log: No start time found"
				print printstr
				if self.write_file:
					self.file.write(printstr+"\n")

	def reset(self):
		if self.on:
			printstr = "------------------------------------------------"
			print printstr
			if self.write_file:
				self.file.write(printstr+"\n")
			self.count=1

	def close(self):
		if self.write_file:
			self.file.close()
			self.write_file=False

	def __del__(self):
		if self.write_file:
			self.file.close()
			self.write_file=False

if __name__=="__main__":
	import os
	os.chdir("/Users/Niermann")

	log=Logger(True,True)
	log.open(os.getcwd())
	log.start("Smooth")

	log.start("Blur")
	
	log.info("test info box")
	log.out("test out box")
	log.out("Decontrast")
	log.end()
	log.end()
	a=[3,2,1]
	log.info(3,a,"isadadasdasdasdasdasdasdasdasdasdasdt","a: ",a,3)
	log.end()
	log.reset()

	log.start("test",a)
	log.start("test")
	log.start("test")
	# log.reset()
	log.out("test1")
	log.info("test2")
	log.end()
	log.end()
	log.end()
	log.end()
	log.end()
	log.start("error")
	log.out("stuff")
	log.info("123")
	log.end()
	log.close()