"""
Streams that require more compute resources on AWS than 4 vcpu 
and 30 gb. (Those are the defaults associated with the prefect 2
workpool.)

"""

COMPUTE_EXCEPTIONS = {
    'RS03AXPS-SF03A-3D-SPKIRA301':{
        '365': '8vcpu_48gb'
    },
}