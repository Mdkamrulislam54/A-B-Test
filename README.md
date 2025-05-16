# A-B-Test


A_B_Testing_Process_with_Optimum_Sample_Size_Calculation.ipynb
A_B_Testing_Process_with_Optimum_Sample_Size_Calculation.ipynb_

[ ]
import pandas as pd
import numpy as np
import altair as alt
alt.data_transformers.disable_max_rows()
from datetime import datetime
Pre-test Metrics Calculation
User Activity

[ ]
## Loading the data from a .csv file
data = pd.read_csv("Activity_pretest.csv")
data.head()


[ ]
# User's with daily activity
data.query('activity_level > 0').head()


[ ]
data.activity_level.value_counts().sort_values(ascending=False)


[ ]
data.activity_level.value_counts().sort_index()


[ ]
activity = data.query('activity_level > 0').groupby(['dt', 'activity_level'])["userid"].nunique().reset_index().rename(columns={"userid":"number_of_active_users"})

activity.head(10)


[ ]
alt.Chart(activity).mark_line(size=1).encode(
    alt.X('dt:T', axis=alt.Axis(title = 'Date')),
    alt.Y('number_of_active_users:Q', axis=alt.Axis(title = 'Number of Users')),
    tooltip=['activity_level'],
    color='activity_level:N'
).properties(
    width=600,
    height=400
)

Calculating Daily Active Users
In this dataset, a userid will count towards DAU if their activity_level for that day is not zero.


[ ]
activity = data.query('activity_level > 0').groupby(['dt'])["userid"].nunique().reset_index().rename(columns={"userid":"number_of_active_users"})
activity.head()


[ ]
activity.describe().round(0)

We have 31 Day's Data and the Mean Daily Active Users (DAU) is 30,673 with a Stanard Deviation of 91


[ ]
alt.Chart(activity).mark_line(size=4).encode(
    alt.X('dt:T', axis=alt.Axis(title = 'date')),
    alt.Y('number_of_active_users:Q', axis=alt.Axis(title = 'Number of Users'))
).properties(
    width=600,
    height=400,
    title='Daily Active Users'
)

Click-through rate

[ ]
## Loading the data from a .csv file
data1 = pd.read_csv("Ctr_pretest.csv")

[ ]
data1.head()


[ ]
data1.describe().round(2)

So, before designing and activating the test we had the Mean Click Through Rate (CTR) 33.0% with a Standard Deviation of 1.73.

So, the Minimum Detection Effect (MDE) should be greater than 1.73% ~ 2.0% in our test's Mean CTR


[ ]
ctr = data1.groupby(['dt'])["ctr"].mean().reset_index().rename(columns={"ctr":"avg_daily_ctr"})
ctr.head()


[ ]
alt.Chart(ctr).mark_line(size=4).encode(
    alt.X('dt:T', axis=alt.Axis(title = 'Date')),
    alt.Y('avg_daily_ctr:Q', axis=alt.Axis(title = 'CTR'), scale=alt.Scale(domain=[32, 34])),
    tooltip=['avg_daily_ctr'],
).properties(
    width=600,
    height=400,
    title='Average Daily CTR'
)

Sample Size Determination and Power Calculation
How many users need to be exposed to the test and how long should we run the test?


[ ]
from scipy import stats
For Binomial Distribution [CTR: Clicked/Not Clicked]

[ ]
def binomial_sample_size(metric, mde, alpha, beta):
    # standard normal distribution to determine z-values
    snd = stats.norm(0, 1)

    Z_beta = snd.ppf(1-beta)
    print(Z_beta)

    Z_alpha = snd.ppf(1-alpha/2)
    print(Z_alpha)

    # average of probabilities from both groups
    p = (metric + metric + mde) / 2
    print(p)

    N = (2 * p * (1 - p) * (Z_alpha + Z_beta)**2 / mde**2)

    return N

[ ]

binomial_sample_size(metric=0.33, mde=0.02, alpha=0.05, beta=0.2) # metric ~ avg_daily_ctr; #mde = 2%; alpha = 5% & beta = 20%

0.8416212335729143
1.959963984540054
0.34
np.float64(8806.443061939677)
So, at least 8,807 users need to be exposed to the test

For Continuous Distribution [Daily Active Users (DAU)]

[ ]
def continuos_sample_size(metric, mde, sd, alpha, beta):
    # standard normal distribution to determine z-values
    snd = stats.norm(0, 1)

    Z_beta = snd.ppf(1-beta)
    print(Z_beta)

    Z_alpha = snd.ppf(1-alpha/2)
    print(Z_alpha)

    N = (2 * sd**2 * (Z_beta + Z_alpha)**2 / mde**2)

    return N

[ ]
continuos_sample_size(metric=30673, mde=300, sd=91, alpha=0.05, beta=0.2)
# metric ~ avg number of daily active users #mde = it should be greater than 91. We are taking 300; ~1% Increase.
0.8416212335729143
1.959963984540054
np.float64(1.4443682906698845)
So, the test should run for at least 1.44 ~ 2 days

A/B Testing Process
Let's now proceed with the A/B Testing:

Assignements
We first need to randomly assign the test to 8,807 Users.

We have the Assignment Data in the "Assignment.csv". We will load the dataset and inspect whether the assignment was done properly between the two groups.


[ ]
data = pd.read_csv("Assignments.csv")
data.head()


[ ]
print(datetime.strptime(data.head(1)['ts'][0], '%Y-%m-%dT%H:%M:%SZ').strftime("%Y-%m-%d"))
2021-11-02

[ ]
data['dt'] = data['ts'].map(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ').strftime("%Y-%m-%d"))
data.head()


[ ]
data.describe().round(4)


[ ]
data.groupby(['groupid'])["userid"].nunique().reset_index().rename(columns={"userid":"number_of_users"})


[ ]
data_count = data.groupby(['groupid','dt'])["userid"].nunique().reset_index().rename(columns={"userid":"number_of_users"})
data_count.head()


[ ]
alt.Chart(data_count).mark_line(size=3).encode(
    alt.X('dt:T'),
    alt.Y('number_of_users:Q'),
    color='groupid:O',
    tooltip=['number_of_users']
).properties(
    width=600,
    height=400
)

Comparing Activity (activeness and activity_level) between the Groups:

[ ]
data_act = pd.read_csv("Activity_all.csv")
data_act.head()


[ ]
data_act.groupby(['groupid','dt']).describe().round(2)

We can clearly observe that the Mean and Median Activity Level of the Test Group (who were exposed to the new ads) is way higher than the Control Group (who were not exposed to the new ads)


[ ]
data_act.query('activity_level > 0').groupby(['dt', 'groupid'])['userid'].nunique().reset_index().rename(columns={"userid":"number_of_active_users"}).head()



[ ]
alt.Chart(data_act.query('activity_level > 0').groupby(['dt', 'groupid'])['userid'].nunique().reset_index().rename(columns={"userid":"number_of_active_users"})).mark_line(size=3).encode(
    alt.X('dt'),
    alt.Y('number_of_active_users'),
    color='groupid:O',
    tooltip=['number_of_active_users']
).properties(
    width=900,
    height=600
)

So, we can see that after the test starts (1st of November, 2021) the test group (groupid = 1) has way more Number of Active Users. This is good for the business. The new adds aren't driving away our active users.

Control Group's Active User's Statistics after the Test Starts:


[ ]

 (
    data_act.query('activity_level > 0 and groupid == 0 and dt >= "2021-11-01"')
    .groupby(['dt','groupid']).count().reset_index()[['groupid','activity_level']].describe().round(2)
)

Test Group's Active User's Statistics after the Test Starts:


[ ]
(
    data_act.query('activity_level > 0 and groupid == 1 and dt >= "2021-11-01"')
    .groupby(['dt','groupid']).count().reset_index()[['groupid','activity_level']].describe().round(2)
)


[ ]
data_act.query('dt >= "2021-11-01"').groupby(['groupid']).describe().round(2)


[ ]
data_act.query('dt < "2021-11-01"').groupby('groupid').describe().round(2)

The Tests
By the Activity Level (Guardrail Metric)

[ ]
#from scipy import stats
from scipy.stats import ttest_ind

[ ]
data_act.query('groupid == 0')['activity_level'].head()


[ ]
data_act.query('groupid == 0')['activity_level'].to_numpy()
array([ 0,  0,  0, ..., 20, 20, 20])
We will run a Two Independent Sample t-tests.

We could have performed the Z (Normal) Test as well as we have a large sample.


[ ]
res = ttest_ind(data_act.query('groupid == 0 and dt >= "2021-11-01"')['activity_level'].to_numpy(),
                data_act.query('groupid == 1 and dt >= "2021-11-01"')['activity_level'].to_numpy()).pvalue

print(res)
0.0

[ ]
"{:.100f}".format(res)

The p-value is very small (<0.05) and hence we can reject the null hypothesis that the Mean Activity Level between the Test and Control Group is Equal at 5% of significance.

By the Number of Active Users (Guardrail Metric)

[ ]
data_act.head()


[ ]
data_act_count = data_act.query('activity_level > 0').groupby(['dt','groupid'])["userid"].nunique().reset_index().rename(columns={"userid":"number_of_active_users"})
data_act_count.head()


[ ]
before = data_act_count.query('dt < "2021-11-01"')

[ ]
before.head()


[ ]
after = data_act_count.query('dt >= "2021-11-01"')

[ ]
after.head()

Checking for the Pre-test Bias on Activity:


[ ]
np.mean(before.query('groupid == 0')['number_of_active_users'].to_numpy())
np.float64(15320.870967741936)

[ ]
np.mean(before.query('groupid == 1')['number_of_active_users'].to_numpy())
np.float64(15352.516129032258)
So, before the test started the Mean Daily Active Users between the groups were similar. So, no pre-test Bias Exists.

But, to be sure we will run the hypothesis test:


[ ]
res = ttest_ind(before.query('groupid == 0')['number_of_active_users'].to_numpy(), before.query('groupid == 1')['number_of_active_users']
                .to_numpy()).pvalue

print(res)
0.1630842353828084

[ ]
"{:.100f}".format(res)

The p-value (>0.05) also suggests that the Mean DAU weren't significantly different at 5% Level of Significance before the test. Hence, no pre-test bias existed

Now, let's test what happens after the test starts:


[ ]
np.mean(after.query('groupid == 0')['number_of_active_users'].to_numpy())
np.float64(15782.0)

[ ]
np.mean(after.query('groupid == 1')['number_of_active_users'].to_numpy())
np.float64(29302.433333333334)
A clear difference between the groups. But, let's perform the hypothesis test to draw inference about the population:


[ ]
res = ttest_ind(after.query('groupid == 0')['number_of_active_users'].to_numpy(), after.query('groupid == 1')['number_of_active_users']
                .to_numpy()).pvalue

print(res)
6.590603584107244e-84

[ ]
"{:.100f}".format(res)

The p-value is very small (<0.05) and hence we can reject the null hypothesis that the Mean Daily Active Users (DAU) between the Test and Control Group is Equal at 5% of significance after the test starts.

Click through rate (CTR) [Success Metric]

[ ]
data_ctr = pd.read_csv("Ctr_all.csv")
data_ctr.head()


[ ]
data_ctr_avg = data_ctr.groupby(['groupid','dt'])["ctr"].mean().reset_index().rename(columns={"ctr":"avg_daily_ctr"})
data_ctr_avg


[ ]
alt.Chart(data_ctr_avg).mark_line(size=5).encode(
    alt.X('dt'),
    alt.Y('avg_daily_ctr'),
    color='groupid:O',
    tooltip=['avg_daily_ctr']
).properties(
    width=600,
    height=400
)

A Clear Increament between the groups after the test started.


[ ]
before = data_ctr.query('dt < "2021-11-01"')[['groupid', 'ctr']]
before


[ ]
after = data_ctr.query('dt >= "2021-11-01"')[['groupid', 'ctr']]
after


[ ]
before.query('groupid == 0')['ctr'].to_numpy().mean()
np.float64(33.00091277553074)

[ ]
before.query('groupid == 1')['ctr'].to_numpy().mean()
np.float64(32.99957172093258)
Before the test the mean CTR was similar.


[ ]
after.query('groupid == 0')['ctr'].to_numpy().mean()
np.float64(32.996977569382835)

[ ]
after.query('groupid == 1')['ctr'].to_numpy().mean()
np.float64(37.99695912626142)
But, after the test started there was a clear improvement in the mean CTR.

Let's compare the Standard Deviations as well.


[ ]
before.query('groupid == 0')['ctr'].to_numpy().std()
np.float64(1.7336979501682888)

[ ]
before.query('groupid == 1')['ctr'].to_numpy().std()
np.float64(1.7296548367391134)

[ ]
after.query('groupid == 0')['ctr'].to_numpy().std()
np.float64(1.7331985918552912)

[ ]
after.query('groupid == 1')['ctr'].to_numpy().std()
np.float64(1.7323710606903675)
Now, we want to prove this point by performing hypothesis tests:

Before the Test:


[ ]
res = ttest_ind(before.query('groupid == 0')['ctr'].to_numpy(), before.query('groupid == 1')['ctr']
                .to_numpy()).pvalue

print(res)
0.705741417344299
High p-value (p > 0.05); Hence, failed to reject the null hypothesis (mean ctr between the groups is equal) at 5% level of significance.

After the Test:


[ ]
res = ttest_ind(after.query('groupid == 0')['ctr'].to_numpy(), after.query('groupid == 1')['ctr']
                .to_numpy()).pvalue
print(res)
0.0

[ ]
"{:.100f}".format(res)

Very Low p-value (p < 0.05); Hence, rejecting the null hypothesis (mean ctr between the groups is equal) at 5% level of significance.

Hence, we can conclude that the new ad policy is a hit interms of our success metric (CTR) as well as our Guardrail Metrics (Daily Active Users and Daily Activity Level).

Colab paid products - Cancel contracts here
