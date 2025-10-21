import pandas as pd
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Financial_inclusion_dataset.csv')

st.markdown("<h1 style = 'color: #114232; text-align: center; font-size: 60px; font-family: Monospace'>FINANCIAL INCLUSION APP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #87A922; text-align: center; font-family: cursive '>Built by Habeeb_Ayobami</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)

st.image('BankIamges.png')

#Add Project proble statement
st.markdown("<h2 style = 'color: #FF9800; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)

st.markdown("Loan prediction involves using historical data on loan applicants to develop predictive models that can assess the creditworthiness of future loan applicants. This typically includes analyzing factors such as income, coapplicant income, and other relevant financial information to determine the likelihood of a borrower repaying a loan. By studying these factors and their impact on loan approval and repayment, financial institutions can make more informed decisions when evaluating loan applications. This helps reduce the risk of default and ensures that loans are granted to individuals who are more likely to repay them.</p>", unsafe_allow_html=True)


st.sidebar.image('Basic_Ui__28186_29.jpg')

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width=True)

Country = st.sidebar.selectbox('Country', data['country'].unique())
Year = st.sidebar.number_input('Year', data['year'].min(), data['year'].max())
Location_Type = st.sidebar.selectbox('Location Type', data['location_type'].unique())
Cellphone_access = st.sidebar.selectbox('Cellphone Access', data['cellphone_access'].unique())
Household_Size = st.sidebar.number_input('Household size', data['household_size'].min(), data['household_size'].max())
Age = st.sidebar.number_input('Age of Respondent', data['age_of_respondent'].min(), data['age_of_respondent'].max())
Gender = st.sidebar.selectbox('Gender of Respondent', data['gender_of_respondent'].unique())
Relationship_With_Head = st.sidebar.selectbox('Relationship with Head', data['relationship_with_head'].unique())
Marital_Status = st.sidebar.selectbox('Marital Status', data['marital_status'].unique())
Education_Level = st.sidebar.selectbox('Education Level', data['education_level'].unique())
Job_Type = st.sidebar.selectbox('Job Type', data['job_type'].unique())


#users input
input_var = pd.DataFrame()
input_var['country'] = [Country]
input_var['year'] = [Year]
input_var['location_type'] = [Location_Type]
input_var['cellphone_access'] = [Cellphone_access]
input_var['household_size'] = [Household_Size]
input_var['age_of_respondent'] = [Age]
input_var['gender_of_respondent'] = [Gender]
input_var['relationshiop_with_head'] = [Relationship_With_Head]
input_var['marital_status'] = [Marital_Status]
input_var['education_level'] = [Education_Level]
input_var['job_type'] = [Job_Type]


st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.subheader('User Input')
st.dataframe(input_var, use_container_width=True)


Country = joblib.load('country_encoder.pkl')
Location_Type = joblib.load('location_type_encoder.pkl')
Cellphone_access = joblib.load('cellphone_access_encoder.pkl')
Gender = joblib.load('gender_of_respondent_encoder.pkl')
Relationship_With_Head = joblib.load('relationship_with_head_encoder.pkl')
Marital_Status = joblib.load('marital_status_encoder.pkl')
Education_Level = joblib.load('education_level_encoder.pkl')
Job_Type = joblib.load('job_type_encoder.pkl')

#Bring in the model
model = joblib.load('bank_account_encoder.pkl')

predict = model.predict(input_var)


if st.button('Check Your Bank Account Status'):
    if predict[0] == 0:
        st.error(f"This Customer is not likely to have a bank account.")
    else:
        st.success(f"This Customer is likely to have a bank account.")