
import numpy as np
DEFAULT_SCALE = {"A+":97, "A":94, "A-":90, "B+":87, "B":83, "B-":80, "C+":77, "C":73}


def display(score, course_name, scale=DEFAULT_SCALE):
	show_score = str(round(score,2))+"%"
	lower_bound = 0
	letter_grade="D"
	for key,val in sorted(scale.items()):
		if val<score and score-val <= 4:
			letter_grade = "an "+key if key[0]=='A' else "a "+key
			lower_bound = val
			break
	print(f"Your {course_name} course has {show_score} in total, which is {letter_grade}({lower_bound}%)")


def is_valid_format(list):
	if len(list)%2 != 0:
		return False
	for i in range(0, int(len(list)/2)):
		my_score = list[2*i]
		full_score = list[2*i+1]
		if my_score > full_score:
			return False
	return True

# dict - {"Homework":[97,100]}
# proportion - must be an np.array, ie. [.75, .25]
def score_calc(dict, proportion, in_score=False):
	means = []
	index = 0
	for key, val in sorted(dict.items()):
		if not is_valid_format(val):
			print("Value format incorrect.")
			return
		mean = calc_mean(val, in_score)
		means.append(mean)
	sum = 0
	for i in range(0,len(proportion)):
		sum += means[i]*proportion[i]
	result = 100*sum/np.sum(proportion)
	return result


def calc_mean(score_list, in_score=False):
	if not in_score:
		percentage_list=np.array([])
		for i in range(0, int(len(score_list)/2)):
			percentage = score_list[2*i]/score_list[2*i+1]
			percentage_list = np.append(percentage_list, percentage)
		return np.average(percentage_list)
	else:
		all_full = 0
		all_mine = 0
		for i in range(0, len(score_list/2)):
			all_mine += score_list[i]
			all_full += score_list[2*i+1]
		return all_mine / all_full


# probability_scores = {'Homework':[49,50, 91,100, 147,150, 98,100, 100,100, 45,50, 99,100],
# 	"Midterm1":[97,100]}
# probability_proportion = np.array([.25, .40])
# probability_scale = DEFAULT_SCALE
# display(score_calc(probability_scores, probability_proportion),
# 	"Probability", probability_scale)

algorithm_scores = {'Homework':[38,38, 42,45, 65,70, 39,40, 16,18, 47,54, 24,25, 16,22, 35.5,42, 30,35,  # 1-10
								23,30, 52,57],
	"Midterm1": [51.5,65],
	#"Midterm2": [65,65],
	#"Final": [100,100]
	}
# Interestingly, participation is "\epsilon"
algorithm_proportion = np.array([.50, .15])
algorithm_scale = DEFAULT_SCALE
display(score_calc(algorithm_scores, algorithm_proportion),
	"Algorithms", algorithm_scale)