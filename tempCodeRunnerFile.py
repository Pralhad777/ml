# create new data for predictions
new_data = [[70, 5, 8], [55, 18, 5], [39, 4, 6]]

# make predictions on new data
for matrics in new_data: 
    matrics = np.array(matrics)
    matrics = matrics.reshape(-1,3)
    predictions = svc.predict(matrics)

    #print the predictions
    print(matrics)
    print("Average Score: ", matrics[0][0])
    print("Absence: ", matrics[0][1])
    print("Study Hour: ", matrics[0][2])

    predictions = str(predictions)
    length = len(predictions)
    predictions1 = predictions[2:(length-2)]
    predictions1 = '\t=> ' + predictions1
    print("Follow the following advises")

    predictions1 = predictions1.replace(".", ".\n\t=>")
    predictions1 = ",\n\t=>".join(predictions1.split(","))

    length = len(predictions1)
    if predictions1[length-1] == '>':
        predictions1 = predictions1[:length-2]
        
    print(predictions1)