import scipy.io  						#Import scipy.io to read .mats
import scipy.optimize as opt			#Import scipy.optimize to do minimization
import scipy.stats as dists				#Import stats for the normal distribution
import numpy as np 						#Import numpy
import matplotlib.pyplot as plt 		#Import plot tools	

#Purpose: import a .mat (matlab file) and make into an array that we can analyze
def read_mat(mat_name):
	mat = scipy.io.loadmat(mat_name)	#Load the .mat as a dictionary
	data = mat[mat_name[0:-4]] 			#Get the data from the dictionary
	return data 						#Return the array

#Purpose: get all of the contrasts from the data
def get_all_contrasts(data):
	all_contrasts = data[:,0]	#Get all of the contrasts
	return all_contrasts		#Return the contrast array

#Purpose: get the unique contrasts (ascending order) from the data as a list
def get_unique_contrasts(all_contrasts):
	unique_contrasts = list(set(all_contrasts))	#Get the unique contrasts
	unique_contrasts.sort()						#Sort the contrasts
	return unique_contrasts 					#Return the contrasts

#Purpose: get all responses from the data
def get_responses(data): 				
	all_responses = data[:,1]	#Get all of the responses  				
	return all_responses		#Return the response array

#Purpose: get the percent correct for each contrast as a list
def get_percent_correct(responses, all_contrasts):
	unique_contrasts = get_unique_contrasts(all_contrasts)					#Get the unique contrasts
	percent_correct = []													#Set up an empty list for the percent correct
	for contr in unique_contrasts:											#Iterate over the unique contrasts
		contr_idx = [i for i, j in enumerate(all_contrasts) if j == contr]	#Get the indicies for the current contrasts
		#Enumerate returns a list of tuples (i,j) where i is the index and j is the value, the above list comprehenstion
		#iterates over all contrasts and returns the indices relevant to the current contrast
		these_responses = responses[contr_idx]								#Get the responses for the current contrast 
		num_responses = len(these_responses)								#Get the number of responses
		num_corr_responses = sum(these_responses)							#Get the number of correct responses
		pc = num_corr_responses / num_responses 							#Calculate the percent correct
		percent_correct.append(pc) 											#Add the percent correct the list
	return percent_correct 													#Return the percent correct list

#Purpose: signal detection model for percent correct as a function of contrast
def sdt_model(contrast, alpha, beta):
	d_prime = (contrast / alpha) ** beta	#Calculate d'
	pc = dists.norm.cdf(0.5 * d_prime)		#Calculate percent correct
	return pc 								#Return the percent correct

#def safe_ln(x, minval=0.0000000001):
#    return np.log(x.clip(min=minval))	

#Purpose: calculate the negative log likelihood for the signal detection model
def negative_log_likelihood(params, all_contrasts, responses):
	alpha = params[0]															#Get alpha
	beta = params[1]															#Get beta
	correct_trial_idx = [i for i, j in enumerate(responses) if j == 1]			#Get the correct trial indicies
	correct_trials = all_contrasts[correct_trial_idx]							#Get the contrasts for the correct trials
	incorrect_trial_idx = [i for i, j in enumerate(responses) if j == 0]		#Get the incorrect trial indicies
	incorrect_trials = all_contrasts[incorrect_trial_idx]  						#Get the contrasts for the incorrect trials
	like = 0;																	#Set the likelihood to 0
	add_correct = sum(np.log(sdt_model(correct_trials, alpha, beta)))			#Calculate the log likelihood of the correct data
	add_incorrect = sum(np.log(1 - sdt_model(incorrect_trials, alpha, beta)))	#Calculate the log likelihood of the incorrect data
	like = like + add_correct + add_incorrect;									#Add up the log likelihoods
	nll = -like 																#Get the negative of the log likelihood 
	return nll 																	#Return the negative log likelihood

#Purpose: fit the signal detection model to the data
def fit_sdt_model(init_alpha, init_beta, all_contrasts, responses):
	x0 = np.array([init_alpha, init_beta])										#Set up the initial parameter estimates
	xopt = opt.fmin(negative_log_likelihood, x0, (all_contrasts, responses))	#Perform the minimization
	return xopt																	#Return the optimal parameters

#Purpose: main method to analyzing psychophysical data
def analyze_data(filename, init_alpha, init_beta):
	#init_alpha = 0.038283905330556
	#init_beta = 1.337064476382359
	data = read_mat(filename) 												#Read in the data file
	all_contrasts = get_all_contrasts(data) 								#Get the contrasts
	responses = get_responses(data) 										#Get the responses
	unique_contrasts = get_unique_contrasts(all_contrasts) 					#Get the unique contrasts
	percent_correct = get_percent_correct(responses,all_contrasts)			#Get the percent correct for each contrast
	xopt = fit_sdt_model(init_alpha, init_beta, all_contrasts, responses) 	#Fit a model to the psychophysical data
	model_contrasts = np.linspace(0.01, 0.15, 100) 							#Get a list of contrasts for the model
	model_pc = sdt_model(model_contrasts, xopt[0], xopt[1])					#Calculate the percent correct for the model over those contrasts
	plt.semilogx(model_contrasts, model_pc,'b') 							#Plot the model and the data
	plt.semilogx(unique_contrasts, percent_correct, 'bo')
	plt.xlabel('Contrast')
	plt.ylabel('Proportion Correct')
	plt.show()
	return xopt 															#Return the model parameters