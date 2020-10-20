#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
import sagemaker
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

# role = get_execution_role()
role = 'arn:aws:iam::533155507761:role/service-role/AmazonSageMaker-ExecutionRole-20190312T160681 '


# In[ ]:





# In[24]:


inputs = 'file://emnist_train/'


# In[25]:





# In[31]:


from sagemaker.pytorch import PyTorch
emnist_estimator = PyTorch(entry_point='main.py',
                             role=role,
                             framework_version='1.5.0',
                             py_version='py3',
                             training_steps=1000, 
                             evaluation_steps=100,
                             instance_count=1,
                             instance_type='local')

emnist_estimator.fit(inputs)


# In[ ]:




