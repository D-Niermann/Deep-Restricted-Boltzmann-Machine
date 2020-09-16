

if len(additional_args)>0:
	log.out("Loading Settings from ", additional_args[0])
	Settings = __import__(additional_args[0])
else:
	import DefaultSettings as Settings

# rename 
UserSettings = Settings.UserSettings