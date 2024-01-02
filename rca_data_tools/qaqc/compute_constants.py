"""
Streams that require more compute resources on AWS than 4 vcpu 
and 30 gb. (Those are the defaults associated with the prefect 2
workpool.)

"""

COMPUTE_EXCEPTIONS = {
    # spkira
    'CE04OSPS-SF01B-3D-SPKIRA102':{
        '365': '4vcpu_30gb',
        '30': '4vcpu_30gb',
        '7': '4vcpu_30gb',
        '1': '8vcpu_30gb',
    },
    'RS01SBPS-SF01A-3D-SPKIRA101':{
        '365': '4vcpu_30gb',
        '30': '4vcpu_30gb',
        '7': '4vcpu_30gb',
        '1': '4vcpu_30gb',
    },
    'RS03AXPS-SF03A-3D-SPKIRA301':{
        '365': '4vcpu_30gb',
        '30': '4vcpu_30gb',
        '7': '4vcpu_30gb',
        '1': '4vcpu_30gb',
    },
    # velptd
    'RS01SBPS-SF01A-4B-VELPTD102':{
        '365': '4vcpu_30gb',
    },
    'RS03AXPS-SF03A-4B-VELPTD302':{
        '365': '4vcpu_30gb',
    },
    # ctdbpo
    'CE04OSBP-LJ01C-06-CTDBPO108':{
        '365': '4vcpu_30gb',
    }
}
