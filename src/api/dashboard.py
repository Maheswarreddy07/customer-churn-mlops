import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn Dashboard",page_icon="📊",layout="centered")

st.title("📊 Customer Churn Prediction System")
st.markdown("Enter customer details below to evaluate risk status in real-time.")
st.divider()

col1,col2=st.columns(2)

with col1:
    age=st.slider("Customer Age",min_value=18,max_value=100,value=35)
    monthly_charges=st.number_input("Monthly Charges($)",min_value=10.0,max_value=300.0,value=75.0,step=1.0)
    support_tickets=st.slider("Open Support Tickets",min_value=0,max_value=20,value=2)

with col2:
    subscription_plan=st.selectbox("Subscription Tier",options=["Basic","Standard","Premium"],index=1)
    contract_type = st.selectbox("Contract Terms", options=["Month-to-month", "One year", "Two year"], index=0)

    plan_mapping={"Basic":0,"Standard":1,"Premium":2}
    contract_mapping={"Month-to-month":0,"One year":1,"Two year":2}

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Evaluate Churn Risk",type="primary",use_container_width=True):

        payload={
            "Age":float(age),
            "SubscriptionPlan":int(plan_mapping[subscription_plan]),
            "MonthlyCharges":float(monthly_charges),
            "ContractType":int(contract_mapping[contract_type]),
            "SupportTickets":float(support_tickets)
        }

        try:
            backend_url="http://127.0.0.1:8000/predict"
            response=requests.post(backend_url,json=payload)

            if response.status_code==200:
                result=response.json()
                prob=result["churn_probability"]
                status=result["status"]

                st.markdown("### 🎯 System Analysis Evaluation")

                if result["churn_prediction"]==1:
                    st.error(f"⚠️ **ALERT: {status}**")
                    st.metric(label="Churn Probability",value=f"{prob*100:.0f}%",delta="HIGH RISK",delta_color="inverse")
                    st.warning("Recommendation:Dispatch retention discount incentive or support check-in immediately.")
                else:
                    st.success(f"✅ **SAFE:{status}**")
                    st.metric(label="Churn Probability",value=f"{prob*100:.0f}%",delta="LOW RISK")

            else:
                st.error(f"Error communicating with prediction server.code:{response.status_code}")


        except Exception as e:
            st.error(f"Could not connect to FastAPI server backend.Ensure app.py is actively running!Error:{e}")