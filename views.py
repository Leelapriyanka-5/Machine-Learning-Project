from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd

from django.core.files.storage import FileSystemStorage
path=settings.MEDIA_ROOT + "//" + 'Luke_hair_loss.csv'
df = pd.read_csv(path)


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid=request.POST.get("loginid")
        password=request.POST.get("pswd")
        print(loginid)
        print(password)
        try:
            check=UserRegistrationModel.objects.get(loginid=loginid,password=password)
            status=check.status
            if status=="activated":
                request.session['id']=check.id
                request.session['loginid']=check.loginid
                request.session['password']=check.password
                request.session['email']=check.email
                return render(request,'users/UserHome.html')
            else:
                messages.success(request,"your account not activated")
            return render(request,"UserLogin.html")
        except Exception as e:
            print('=======>',e)
        messages.success(request,'invalid details')
    return render(request,'UserLogin.html')
    
def UserHome(request):
    return render(request,"users/UserHome.html",{})



import pandas as pd
from django.shortcuts import render

def hair_data_view(request):
    # Assuming you have a dataset in CSV format named "hair_data.csv"
    # and it is located in the media folder of your Django project.
    path = settings.MEDIA_ROOT + "//" + 'Luke_hair_loss.csv'
    
    # Read the dataset into a pandas DataFrame
    df = pd.read_csv(path)

    # Drop the "hair_loss" column
    df = df.drop(columns=['hair_loss'])
    print("=====================done=========================")

    # Convert the DataFrame to a list of dictionaries for template rendering
    data = df.to_dict(orient='records')

    # Define column names for the template
    columns = df.columns

    return render(request, 'users/dataset.html', {'data': data, 'columns': columns})

import pandas as pd
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from django.conf import settings

#  Define a function to preprocess the data
def preprocess_data(df):
    # Handle null values (you can choose an appropriate strategy)
    df = df.dropna()

    # Apply label encoding to categorical columns if needed
    label_encoder = LabelEncoder()
    df['shampoo_brand'] = label_encoder.fit_transform(df['shampoo_brand'])
    df['stress_level'] = label_encoder.fit_transform(df['stress_level'])
    df['pressure_level'] = label_encoder.fit_transform(df['pressure_level'])
    df['school_assesssment'] = label_encoder.fit_transform(df['school_assesssment'])
    df['dandruff'] = label_encoder.fit_transform(df['dandruff'])
    df['swimming'] = label_encoder.fit_transform(df['swimming'])
    df['hair_washing'] = label_encoder.fit_transform(df['hair_washing'])
    
    return df

# Define a function to train the K-NN classifier
def knn_classification(request):
    # Assuming you have a dataset in CSV format named "hair_data.csv"
    # and it is located in the media folder of your Django project.
    path = settings.MEDIA_ROOT + "//" + 'Luke_hair_loss.csv'
    
    # Read the dataset into a pandas DataFrame
    df = pd.read_csv(path)

    # Preprocess the data
    df = preprocess_data(df)

    # Split the data into features (X) and the target variable (y)
    X = df[['stay_up_late', 'pressure_level', 'coffee_consumed', 'brain_working_duration',
            'school_assesssment', 'swimming',
            'hair_washing', 'hair_grease', 'dandruff', 'libido']]
    
    y = df['hair_loss']

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Feature scaling (optional but can be beneficial for k-NN)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # Create a k-NN classifier (you can adjust the number of neighbors, 'n_neighbors')
    knn_classifier = KNeighborsClassifier(n_neighbors=5)

    # Train the k-NN classifier on the training data
    knn_classifier.fit(X_train, y_train)

    # Return the trained classifier
    return knn_classifier
# Assuming you have a DataFrame 'df' with your dataset

def prediction(request):
    if request.method == 'POST':
        # Get user input values from the form
        stay_up_late = float(request.POST['stay_up_late'])
        pressure_level = float(request.POST['pressure_level'])
        coffee_consumed = float(request.POST['coffee_consumed'])
        brain_working_duration = float(request.POST['brain_working_duration'])
        school_assesssment = float(request.POST['school_assesssment'])
        swimming = float(request.POST['swimming'])
        hair_washing = float(request.POST['hair_washing'])
        hair_grease = float(request.POST['hair_grease'])
        dandruff = float(request.POST['dandruff'])
        libido = float(request.POST['libido'])

        # Create a DataFrame with the user's input data
        input_data = pd.DataFrame({
            'stay_up_late': [stay_up_late],
            'pressure_level': [pressure_level],
            'coffee_consumed': [coffee_consumed],
            'brain_working_duration': [brain_working_duration],
            'school_assesssment': [school_assesssment],
            'swimming': [swimming],
            'hair_washing': [hair_washing],
            'hair_grease': [hair_grease],
            'dandruff': [dandruff],
            'libido': [libido]
        })

        # Call the knn_classification function to get the trained K-NN classifier
        knn_classifier = knn_classification(request)

        # Predict hair loss for the user's input
        predicted_hair_loss = knn_classifier.predict(input_data)

        # Rest of your code remains the same
        print('--------------------')
        print(predicted_hair_loss)
        print(type(predicted_hair_loss))
        print(len(predicted_hair_loss))
        # Determine the prevention tips based on the predicted hair loss level
        prevention_tips = ""
        level = ""
        if predicted_hair_loss[0] == "Few":
            level = "Level 1"
            prevention_tips = [
                "Maintain a healthy diet with essential nutrients for hair growth.",
                "Avoid excessive stress and practice relaxation techniques.",
                "Use a mild and suitable shampoo for your hair type.",
                "Avoid excessive use of heat styling tools."
            ]
        elif predicted_hair_loss[0] == "Medium":
            level = "Level 2"
            prevention_tips = [
                "Increase your intake of vitamins and minerals beneficial for hair health.",
                "Consider using hair care products designed for your hair type and needs.",    
            ]
        elif predicted_hair_loss[0] == "Many":
            level = "level 3"
            prevention_tips = [
                "Consult a dermatologist or hair specialist for a thorough assessment.",
                "Explore advanced hair care treatments and therapies.",            
            ]
        elif predicted_hair_loss[0] == "A lot":
            level = "level 4"
            prevention_tips = [
                "Seek immediate professional help from a hair loss expert.",
                "Consider hair restoration treatments if appropriate.",            
            ]
        else:
            return render(request,'users/result.html',{'msg':"No Hair Loss"})
        
        context = {'level':level,"tips":prevention_tips}

        return render(request, 'users/result.html', context)

    else:
        return render(request, 'users/prediction_form.html')






























































# Define a view for predicting hair loss
# def prediction(request):
#     if request.method == 'POST':
#         # Get user input values from the form
#         stay_up_late = float(request.POST['stay_up_late'])
#         pressure_level = float(request.POST['pressure_level'])
#         coffee_consumed = float(request.POST['coffee_consumed'])
#         brain_working_duration = float(request.POST['brain_working_duration'])
#         school_assesssment = float(request.POST['school_assesssment'])
#         swimming = float(request.POST['swimming'])
#         hair_washing = float(request.POST['hair_washing'])
#         hair_grease = float(request.POST['hair_grease'])
#         dandruff = float(request.POST['dandruff'])
#         libido = float(request.POST['libido'])

#         # Create a DataFrame with the user's input data
#         input_data = pd.DataFrame({
#             'stay_up_late': [stay_up_late],
#             'pressure_level': [pressure_level],
#             'coffee_consumed': [coffee_consumed],
#             'brain_working_duration': [brain_working_duration],
#             'school_assesssment': [school_assesssment],
#             'swimming': [swimming],
#             'hair_washing': [hair_washing],
#             'hair_grease': [hair_grease],
#             'dandruff': [dandruff],
#             'libido': [libido]
#         })

#         # Feature scaling for the user's input data
#         scaler = StandardScaler()
#         input_data_scaled = scaler.fit_transform(input_data)

#         # Call the knn_classification function to get the trained K-NN classifier
#         knn_classifier = knn_classification(request)
        

#         # Predict hair loss for the user's input
#         predicted_hair_loss = knn_classifier.predict(input_data_scaled)

#         return render(request, 'users/result.html', {
#             'predicted_hair_loss': predicted_hair_loss[0]
#         })

#     else:
#         return render(request, 'users/prediction_form.html')