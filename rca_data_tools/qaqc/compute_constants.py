"""
Streams that require more compute resources on AWS than 4 vcpu 
and 30 gb. (Those are the defaults associated with the prefect 2
workpool.)

"""

COMPUTE_EXCEPTIONS = {
    'RS03AXPS-SF03A-3D-SPKIRA301':{
        '365': '8vcpu_48gb'
    },
    'RS01SBPS-SF01A-3D-SPKIRA101':{
        '365': '8vcpu_48gb',
        '30': '8vcpu_48gb',
        '7': '8vcup_48gb',
    },
}