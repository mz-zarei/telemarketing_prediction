# Public Term Deposit Conversion Prediction Model
The goal of this project is to predict which customers are likely to subscribe to a term deposit product at a financial institution, based on data from past direct marketing campaigns. The campaigns were conducted via phone calls, and the data includes information about the customer's demographic and financial characteristics, as well as details about the marketing campaign itself.

The dataset contains the following fields:


1. age: The age of the customer (numeric)
2. job: The type of job the customer has (categorical: 'admin.' 'blue-collar', 'entrepreneur','housemaid','management', 'retired' 'self-employed','services','student','technician','unemployed','unknown')
3. marital: The marital status of the customer (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4. education: The education level of the customer (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course', 'university.degree', 'unknown')
5. default: Does the customer have credit in default? (categorical: 'no','yes','unknown')
6. housing: Does the customer have a housing loan? (categorical: 'no','yes','unknown')
7. loan: Does the customer have a personal loan? (categorical: 'no','yes','unknown')
8. contact: The type of contact communication used during the campaign (categorical: 'cellular','telephone')
9. day_of_week: The day of the week of the last contact (categorical: 'mon','tue','wed','thu','fri')
10. duration: The duration of the last contact, in seconds (numeric). Note: this attribute highly affects the output target (e.g., if duration=0 then y='no')
11. campaign: The number of contacts performed during this campaign and for this client (numeric, includes last contact)
12. pdays: The number of days that have passed since the customer was last contacted from a previous campaign (numeric; 999 means the customer was not previously contacted)
13. previous: The number of contacts performed before this campaign and for this client (numeric)
14. emp.var.rate: Employment variation rate - quarterly indicator (numeric)
15. cons.price.idx: Consumer price index - monthly indicator (numeric)
16. cons.conf.idx: Consumer confidence index - monthly indicator (numeric)
17. euribor3m: Euribor 3 month rate - daily indicator (numeric)
18. nr.employed: Number of employees - quarterly indicator (numeric)
19. poutcome: The outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
20. month: The month of the last contact (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')

The target variable, or desired output, is:
21. y: Has the client subscribed a term deposit? (binary: 'yes','no')





