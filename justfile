record:
	#!/usr/bin/env python3
	import random
	import os
	import time	
	def	tag_random(vowel):
		return './traindat_staging/{}_{}.wav'.format(
			vowel,
			str(int(random.random() * 10000))
		)
	if not os.path.exists('traindat_staging'):
		os.makedirs('traindat_staging')
	vowels = ["a", "e", "i", "o", "u"]
	for v in vowels:
		filename = tag_random(v)
		cmd = ['rec', '-c', '1', filename, 'trim', '0', '1']
		print(v, '...')
		time.sleep(2)
		os.system(' '.join(cmd))
check:
	play traindat_staging/*
approve:
	mv traindat_staging/* traindat
reject:
	rm traindat_staging/*

go:
	pipenv run python ./go.py testdat/a_4525.wav
