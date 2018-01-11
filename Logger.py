# -*- coding: utf-8 -*-
""" 
	Dario Niermann
	    2017

"""


class Logger(object):
	def __init__(self,on):
		self.count=1
		self.count_save=[]
		self.start_time=[0]
		self.end_time=0
		self.process_name=[""]
		self.on=on #True false


	def info(self,*args):
		if self.on:			
			string="Info     "+"\t"*(len(self.start_time)-1)
			for i in args:
				string+=str(i)+" "
			print string 


	def out(self,input_str,*args):
		if self.on:
			if self.count==1:
				print ""
			for i in args:
				input_str+=" "+str(i)
			try:
				print str(self.count)+".      ","\t"*(len(self.start_time)-1),str(input_str)
				self.count+=1
			except:
				print "Error while logging"
		
	def start(self,input_str,*args):
		if self.on:
			if self.count==1:
				print ""
			for i in args:
				input_str+=" "+str(i)
			from time import time
			try:
				start_len=(len(self.start_time))
				print str(self.count)+".","-"*start_len+"v"+" "*(5-start_len)+"\t"*(len(self.start_time)-1),str(input_str)
				self.count_save.append(self.count)
				self.count+=1
				self.start_time.append(time())
				self.process_name.append(input_str)
			except:
				print "Error: while begin logging"

	def end(self):
		if self.on:
			if self.start_time[len(self.start_time)-1]!=0:
				from time import time
				self.end_time=time()
				try:
					start_len=(len(self.start_time)-1)
					print str(self.count_save.pop())+".","-"*start_len+"^"+" "*(5-start_len)+"\t"*(len(self.start_time)-2),self.process_name[len(self.process_name)-1]," (took "+str(round(self.end_time-self.start_time[len(self.start_time)-1],1))+" sek)"
					self.start_time.pop()
					self.process_name.pop()
					self.end_time=0
				except:
					print "Error: while end logging"
			else:
				print "Error : end log: No start time found"

	def reset(self):
		if self.on:
			print "------------------------------------------------"
			self.count=1

if __name__=="__main__":

	log=Logger(True)

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
