import numpy as np


eigenvalue_data = np.asarray([36.11, 41.34, 54.88, 60.28, 64.57,70.46])
intensity_data = np.asarray([7.7, 6.0, 4.1, 7.1, 4.6, 17.0])


def get_data(case_number):

	if case_number == 1:

		#Case 1
		true_ind1 = [0,1,2]
		true_ind2 = [3,4,5]

	elif case_number == 2:	

		#Case 2
		true_ind1 = [1,2,3]
		true_ind2 = [0,4,5]

	elif case_number == 3:	

		#Case 3
		true_ind1 = [1,2,4]
		true_ind2 = [0,3,5]

	elif case_number == 4:	

		#Case 4
		true_ind1 = [1,2,5]
		true_ind2 = [0,3,4]
		
	elif case_number == 5:	

		#Case 5
		true_ind1 = [0,2,3]
		true_ind2 = [1,4,5]

	elif case_number == 6:	

		#Case 6
		true_ind1 = [0,2,4]
		true_ind2 = [1,3,5]	

	elif case_number == 7:	

		#Case 7
		true_ind1 = [0,2,5]
		true_ind2 = [1,3,4]

	elif case_number == 8:	

		#Case 8
		true_ind1 = [0,1,3]
		true_ind2 = [2,4,5]

	elif case_number == 9:	

		#Case 9
		true_ind1 = [0,1,4]
		true_ind2 = [2,3,5]
		
	elif case_number == 10:	

		#Case 10
		true_ind1 = [0,1,5]
		true_ind2 = [2,3,4]

	elif case_number == 11:	

		#Case 11
		true_ind1 = [2,3,4]
		true_ind2 = [0,1,5]

	elif case_number == 12:	

		#Case 12
		true_ind1 = [2,3,5]
		true_ind2 = [0,1,4]

	elif case_number == 13:	

		#Case 13
		true_ind1 = [2,4,5]
		true_ind2 = [0,1,3]	

	elif case_number == 14:	

		#Case 14
		true_ind1 = [1,3,4]
		true_ind2 = [0,2,5]	

	elif case_number == 15:	

		#Case 15
		true_ind1 = [1,3,5]
		true_ind2 = [0,2,4]	

	elif case_number == 16:	

		#Case 16
		true_ind1 = [1,4,5]
		true_ind2 = [0,2,3]	

	elif case_number == 17:	

		#Case 17
		true_ind1 = [0,3,4]
		true_ind2 = [1,2,5]	

	elif case_number == 18:	

		#Case 18
		true_ind1 = [0,3,5]
		true_ind2 = [1,2,4]	

	elif case_number == 19:	

		#Case 19
		true_ind1 = [0,4,5]
		true_ind2 = [1,2,3]

	elif case_number == 20:	

		#Case 20
		true_ind1 = [3,4,5]
		true_ind2 = [0,1,2]		

	else:
		print('wrong case number!')
		exit()	


	true_eigenvalue1 = eigenvalue_data[true_ind1]
	true_eigenvalue2 = eigenvalue_data[true_ind2]
	true_intensity1 = intensity_data[true_ind1]/intensity_data[true_ind1[-1]]
	true_intensity2 = intensity_data[true_ind2]/intensity_data[true_ind2[-1]]

	return true_eigenvalue1, true_eigenvalue2, true_intensity1, true_intensity2


# true_eigenvalue1, true_eigenvalue2, true_intensity1, true_intensity2 = get_data(1)
# print(true_eigenvalue1)




