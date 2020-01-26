import itertools
import json
import os

norm = "LHSCTMRP"


def get_all_nargs():
	total = []
	for i in range(len(norm) + 1):
		total += list(itertools.combinations(norm, i + 1))
	return total + [('', '')]


def handle_features_input():
	print("------------------------------------------")
	features = []
	choose = input("Do you want to use N-Grams in your project? enter y / n: ")
	if choose.lower() == 'y':
		while True:
			type = input("Enter the type of the new N-Grams (c for chars and w for words): ").lower()
			n = input("Enter the N of the N-Grams (1 for Unigrams, 2 for Bigrams etc.): ")
			skips = input("Enter the size of the skips: ")
			count = input("Enter the number of words / chars to count (100 for the top 100 words etc.): ")
			tfidf = input("Do you use TF-IDF (the default is TF)? enter y / n ")
			if tfidf.lower() == 'y':
				tfidf = "tfidf"
			else:
				tfidf = 'tf'
			features.append("ngrams_{}_{}_{}_{}_{}".format(count, type, tfidf, n, skips))
			if input("Do you want to add more N-Grams features in your project? enter y / n: ").lower() != 'y':
				break
	return list(set(features))


def handle_methods_input():
	print("------------------------------------------")
	methods = []
	print("The supported ML methods in the software is MLP, SVC, RF, LR, MNB, RNN")
	while True:
		method = input("Choose one of the ML method from above: ")
		if method.lower() in ['mlp', 'svc', 'rf', 'lr', 'mnb', 'rnn']:
			methods.append(method.lower())
		else:
			print("Please enter the method correctly!")
			continue
		if input("Do you want to add another ML method? enter y / n: ") != 'y':
			break
	return list(set(methods))


def handle_measure_input():
	print("------------------------------------------")
	measure = ['accuracy_score']
	if input("Do you want to use in more measure except from Accuracy? enter y / n: ").lower() == 'y':
		dic = {'1': 'f1_score',
			   '2': 'precision_score',
			   '3': 'recall_score',
			   '4': 'roc_auc_score',
			   '5': 'confusion_matrix',
			   '6': 'roc_curve',
			   '7': 'precision_recall_curve'}
		print("The supported measures in the software is")
		for key, val in dic.items():
			print(str(key) + ' - ' + str(val).replace('_', ' '))
		while True:
			choose = input("Enter the number of the selected measure: ")
			if choose in dic.keys():
				measure.append(dic[choose])
			else:
				print("Please enter number from the list!")
				continue
			if input("Do you want to add another measure? enter y / n: ") != 'y':
				break
	return list(set(measure))


def handle_stylistic_features_input():
	dic = {'cc': "chars count",
                           'wc': "words count",
                           'sc': "sentence count",
                           'emc': "exclamation mark (!) count",
                           'qsmc': "question mark (?) count",
                           'scc': "special characters (@, #, $, &, *, %, ^) count",
                           'qtmc': "quotation mark (\", ') count",
                           'alw': "average letters in words",
                           'als': "average letters in sentence",
                           'aws': "average words in sentence",
                           'awl': "average words length",
                           'ie': "increasing expressions",
                           'de': "decreasing expressions",
                           'nw': "negative terms",
                           'pw': "positive terms",
                           'te': "time expressions",
                           'de': "doubt expressions",
                           'ee': "emotion expressions",
                           'fpe': "first person expressions",
                           'spe': "second person expressions",
                           'tpe': "third person expressions",
                           'ine': "inclusion expressions",
                           'p1': "expressions form power 1",
                           'p2': "expressions form power 2",
                           'p3': "expressions form power 3",
                           'pm1': "expressions form power -1",
                           'pm2': "expressions form power -2",
                           'pm3': "expressions form power -3",
                           'pm4': "expressions form power -4",
                           'ap': "expressions form all the powers",
                           'topa1': "Enable all features on the 1'st trimester of the text",
                           'topa2': "Enable all features on the 2'nd trimester of the text",
                           'topa3': "Enable all features on the 3'rd trimester of the text",
                           'topb1': "Enable all features on the first ten words of the text",
                           'topb2': "Enable all features on the text without 10 first and last 10 words",
                           'topb3': "Enable all features on the last ten words of the text"}
	stylistic_features = []
	print("------------------------------------------")
	if input("Do you want to use stylistic features in your project? enter y / n: ").lower() == 'y':
		print("The supported measures in the software is")
		for key, val in dic.items():
			print(str(key) + ' - ' + str(val))
		while True:
			choose = input("Enter the acronyms of the selected feature: ")
			if choose in dic:
				stylistic_features.append(choose)
			else:
				print("Please enter acronyms from the list!")
				continue
			if input("Do you want to add another feature? enter y / n: ") != 'y':
				break
	return list(set(stylistic_features))


def handle_normalization_input():
	print("------------------------------------------")
	norml = ''
	dic = {'c': 'Spelling Correction',
		   'l': 'Lowercase',
		   'h': 'HTML tags',
		   'p': 'Punctuations',
		   'r': 'Repeated chars',
		   't': 'Stemming',
		   'm': 'Lemmatizer',
		   'e': 'English stop words',
		   'h': 'Hebrew stop words',
		   'x': 'Extended Hebrew stop words'}
	if input("Do you want to normalize the text before the features extraction? enter y / n: ").lower() == 'y':
		print("The supported normalization in the software is")
		for key, val in dic.items():
			print(str(key) + ' - ' + str(val))
		while True:
			choose = input("Enter the selected normalization: ").lower()
			if choose in dic:
				norml += choose
			else:
				print("Please enter normalization from the list!")
				continue
			if input("Do you want to add another normalization? enter y / n: ") != 'y':
				break
	return norml



if __name__ == '__main__':
	train = input("Enter training path:\n")
	test = input("Enter testing path:(optional)\n")
	output_path = input("Enter output path:\n")
	features = handle_features_input()
	methods = handle_methods_input()
	measure = handle_measure_input()
	stylistic_features = handle_stylistic_features_input()
	norm = handle_normalization_input()
	print("------------------------------------------")
	results = input("Enter results path:\n")
	write_dir = input("Enter where to dump all config files:\n")
	nargs = get_all_nargs()
	i = 1
	for n in nargs:
		data = {"train": train, "test": test, "output_csv": output_path, "nargs": ''.join(sorted(list(n))), "features": features,
				"results": results, "methods": methods, "measure": measure, "stylistic_features": stylistic_features}
		name = write_dir + os.sep +"config" + str(i) + ".json"
		with open(name, "w+") as file:
			json.dump(data, file, indent=4)
		i += 1
