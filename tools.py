from sklearn.externals import joblib
model = joblib.load('salary_model.pk1')

print("\t\t\tWelcome to future tools")
print("\t\t\t-----------------------")
print()

while True:
	print('press 1: To estimate salary')
	print('press 2: To estimate salary')
	print('press 3: To estimate salary')
	
	print('Enter your choice : ' , end = '')
	ch = input()
	if int(ch) == 1:
		print('Years of Expirence:' , end = '')
		exp = input()
		print('Estimated Salary :',model.predict([[int(exp)]])[0])
	else:
		print('option not supported now')