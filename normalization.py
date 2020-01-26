import json
import os

import re
import shutil
import string

from nltk import WordNetLemmatizer, SnowballStemmer

from global_parameters import print_message,GlobalParameters
from stopwords_and_lists import stopwords
from stopwords_and_lists import hebrew_stopwords
from stopwords_and_lists import hebrew_stopwords_ex
from autocorrect import spell
from system_config import DICTIONARY_PATH


glbs = GlobalParameters()

def init_tools():
	global dictionary
	dictionary = Dictionary()
	global lemmatizer, stemmer
	lemmatizer = WordNetLemmatizer()
	stemmer = SnowballStemmer("english")


class Dictionary:
	def __init__(self):
		with open(DICTIONARY_PATH, "r") as f:
			self.words = json.load(f)


# region helpful function
import global_parameters


def is_nargs_empty(nargs):
	nargs = nargs.upper()
	for char in string.ascii_uppercase:
		if char in nargs:
			return False
	return True


def remove_repeated_chars(word, index=0):
	if is_word(word):
		if word.lower() in dictionary.words:
			return word
		else:
			while index < len(word) - 1:
				rep, length = have_repeated_chars(word, index)
				if rep is not None:
					while length > 3:
						word = remove_chars(word, rep)
						length -= 1
					new_word = remove_chars(word, rep)
					new_word = remove_repeated_chars(new_word)
					if new_word is not None:
						return new_word
					if index < rep:
						index = rep
					else:
						index += 1
				else:
					return None
		return None
	else:
		rep, _ = have_repeated_chars(word)
		if rep is not None:
			new_word = remove_chars(word, rep)
			return remove_repeated_chars(new_word)
		return word


def remove_chars(word, index):
	new_word = word[:index]
	new_word += word[index + 1:]
	return new_word


def have_repeated_chars(word, idx=0):
	count = 1
	index = None
	max_index = None
	max_count = 0
	for i in range(idx, len(word) - 1):
		if word[i].lower() == word[i + 1].lower():
			if count is 1:
				index = i
			count += 1
		else:
			if count > max_count:
				max_index = index
				max_count = count
				count = 1
	if count > max_count:
		max_index = index
		max_count = count
	return max_index, max_count


def is_word(word):
	for c in word:
		if c not in string.ascii_letters:
			return False
	return True


# endregion

# region Normalization Functions
def remove_html(line):
	regex = r"((<.*)(>|/>))"
	matches = re.findall(regex, line)
	new_line = line
	for match in matches:
		new_line = new_line.replace(match.__getitem__(0), "")

	return new_line


def remove_urls(text):
	return re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)


def remove_stop_words(text):
	buffer = ""
	newword = ""
	for idx, char in enumerate(text):
		if char != ' ' and char not in string.punctuation:
			buffer += char
		else:
			if buffer.lower() not in stopwords:
				newword += buffer
			newword += char
			buffer = ""
	if buffer.lower() not in stopwords:
		newword += buffer
	return newword

def remove_stop_words_hebrew(text):
	text = text.split(' ')
	newword = []
	for word in text:
		for sw in hebrew_stopwords:
			from wordfreq import top_n_list
			if word.startswith(sw) and word[len(sw):] in top_n_list('he', 100000):
				word = word[len(sw):]
				break
		if word not in hebrew_stopwords:
			newword += [word]
	return " ".join(newword)

def remove_stop_words_hebrew_extended(text):
	text = text.split(' ')
	newword = []
	for word in text:
		for sw in hebrew_stopwords_ex:
			from wordfreq import top_n_list
			if word.startswith(sw) and word[len(sw):] in top_n_list('he', 100000):
				word = word[len(sw):]
				break
		if word not in hebrew_stopwords_ex:
			newword += [word]
	return " ".join(newword)


def remove_repetition(text):
	word = ""
	new_line = ""

	for char in text:
		if char != ' ' and char not in string.punctuation:
			word += char
		else:
			new_word = remove_repeated_chars(word)
			if new_word is not None:
				new_line += new_word + " "
			else:
				new_line += word + " "
			if char is not " ":
				new_line += char
			word = ""

	if len(word) > 0:
		new_word = remove_repeated_chars(word)
		if new_word is not None:
			new_line += new_word
		else:
			new_line += word

	return new_line


def spell_correction(text):
	buffer = ""
	new_text = ""

	for char in text:
		if char != ' ' and char not in string.punctuation:
			buffer += char
		else:
			if buffer.lower() not in dictionary.words:
				new_text += spell(buffer) + " "
			else:
				new_text += buffer + " "
			if char != " ":
				new_text += char
			buffer = ""

	if len(buffer) > 0 and buffer.lower() not in dictionary.words:
		new_text += spell(buffer)
	else:
		new_text += buffer

	return new_text


def remove_punctuation(text):
	return text.translate(str.maketrans('', '', string.punctuation))


def lemmatize(text):
	buffer = ""
	new_text = ""

	for char in text:
		if char != ' ' not in string.punctuation:
			buffer += char
		else:
			new_text += lemmatizer.lemmatize(buffer) + " "
			if char != " ":
				new_text += char
			buffer = ""

	if len(buffer) > 0 and buffer.lower() not in dictionary.words:
		new_text += lemmatizer.lemmatize(buffer)
	else:
		new_text += buffer
	return new_text


def stem(text):
	buffer = ""
	new_text = ""

	for char in text:
		if char != ' ' and char not in string.punctuation:
			buffer += char
		else:
			new_text += stemmer.stem(buffer) + " "
			if char != " ":
				new_text += char
			buffer = ""

	if len(buffer) > 0 and buffer.lower() not in dictionary.words:
		new_text += stemmer.stem(buffer)
	else:
		new_text += buffer
	return new_text


def normal(word, nargs):
	if nargs == "" or nargs == "\n":
		return word

	nargs = nargs.lower()
	new_line = word
	if 'l' in nargs:
		new_line = new_line.lower()
	if 'h' in nargs:
		new_line = remove_html(new_line)
	if 'u' in nargs:
		new_line = remove_urls(new_line)
	if 's' in nargs:
		new_line = remove_stop_words(new_line)
	if 'r' in nargs:
		new_line = remove_repetition(new_line)
	if 'c' in nargs:
		new_line = spell_correction(new_line)
	if 'p' in nargs:
		new_line = remove_punctuation(new_line)
	if 't' in nargs:
		new_line = lemmatize(new_line)
	if 'm' in nargs:
		new_line = stem(new_line)

	# addition: 'b' for Hebrew stopwords and 'x' for extended Hebrew stopwords
	if 'x' in nargs:
		new_line = remove_stop_words_hebrew_extended(new_line)
	if 'b' in nargs:
		new_line = remove_stop_words_hebrew(new_line)

	return new_line


# endregion

def normalize(test=False):
	init_tools()
	n = glbs.NORMALIZATION
	if not test:
		i = glbs.TRAIN_DIR
		print_message("Normalizing")
	else:
		i = glbs.TEST_DIR
		print_message("Normalizing test dataset")
	if not is_nargs_empty(n):
		# the normalized folder
		parent_dir = i + "@" + n
		# create the dir if does not exist
		if not os.path.exists(parent_dir):
			os.mkdir(parent_dir)
			for category in os.listdir(i):
				with open(os.path.join(i, category), 'r', encoding='utf8', errors='ignore') as read:
					n_lines = []
					for line in read:
						line = line.rstrip('\n')
						n_lines.append(normal(line, n))
				n_file = '\n'.join(n_lines)
				del n_lines
				# write normalized_file
				with open(os.path.join(parent_dir, category), 'w+', encoding='utf8') as write:
					write.write(n_file)
				del n_file
		else:  # if it does exist
			print_message("found normalized dataset")
			# check if both dirs have the same number of files
			if not len(os.listdir(parent_dir)) == len(os.listdir(i)):
				# delete normalized folder and create a new one.
				print_message("Corrupted normalization found, deleting and starting over...")
				shutil.rmtree(parent_dir, ignore_errors=True)
				normalize()

		glbsNORM_PATH = parent_dir
	else:
		glbsNORM_PATH = i

	return glbsNORM_PATH
