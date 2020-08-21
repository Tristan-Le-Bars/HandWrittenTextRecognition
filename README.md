# HandWrittenTextRecognition
Hand written text recognition ML algorythme implemented with tensorflow2 and trained with the IAM dataset.
This AI recognize the text contained in a word image, it have a word prediction accuracy of ~75% and an error rate of character prediction of ~10%.

# download IAM dataset
- register on the IAM database website : http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
- Download words/words.tgz and extract the files in the data/words/ folder
- Downlead words.txt and put it in the data/ folder


# arguments
```--train``` : train AI until character error rate is no longer reduced for 5 training sessions.

```--make_test``` : use the AI on 115 validation set and display a pyplot window that contain information about the AI performance and make a prediction with the picture Picture/test.png

no argument : train AI until character error rate is no longer reduced for 5 training sessions then use the AI on 115 validation set and display a pyplot window that contain information about the AI performance and make a prediction with the picture Picture/test.png

# demonstration
training :
```bash
Epoch: 1
Batch: 1 / 500 Loss: 2.2806785
Batch: 2 / 500 Loss: 14.285404
Batch: 3 / 500 Loss: 17.222914
Batch: 4 / 500 Loss: 44.94905
...

Set: 1 / 115
succes:  label = "just"  ->  prediction = "just"
failure, 1 mistake(s): label = "go"  ->  prediction = "ao"
succes:  label = "on"  ->  prediction = "on"
failure, 1 mistake(s): label = "on"  ->  prediction = "o"
...

Error rate of character prediciton : 14.786565%
Accuracy of words prediction : 64.800000%
error rate reduced, updating model
Epoch: 2
Batch: 1 / 500 Loss: 1.7570848
Batch: 2 / 500 Loss: 2.7104146
Batch: 3 / 500 Loss: 1.556745
Batch: 4 / 500 Loss: 1.645865
```

validation:

```bash
Set: 1 / 115
succes:  label = "just"  ->  prediction = "just"
succes:  label = "go"  ->  prediction = "go"
failure, 1 mistake(s): label = "on"  ->  prediction = "an"
succes:  label = "and"  ->  prediction = "and"
...

Set: 115 / 115
succes:  label = "Catherine"  ->  prediction = "Catherine"
succes:  label = "said"  ->  prediction = "said"
succes:  label = ","  ->  prediction = ","
succes:  label = "her"  ->  prediction = "her"
...

Error rate of character prediciton : 10.584772%
Accuracy of words prediction : 74.017391%
Predicted word: "test"
```
![alt text](https://raw.githubusercontent.com/Tristan-Le-Bars/HandWrittenTextRecognition/master/doc/pyplot.png)

