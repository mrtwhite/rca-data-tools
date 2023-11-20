import numpy as np 

def select_logger():
    from prefect import get_run_logger
    try:
        logger = get_run_logger()
    except:
        print('Could not start prefect logger...running local log')
        from loguru import logger
    
    return logger


def coerce_qartod_executed_to_int(ds):
    logger = select_logger()

    logger.info(f"ds size pre coercion: {ds.nbytes}")
    qartod_executed_vars = [var for var in ds.variables if 'qartod_executed' in var]
    test_name_list = []
    for var in qartod_executed_vars:
        executed_tests = ds[var].tests_executed.replace(' ', '').split(',')
    
        for i, test in enumerate(executed_tests):
            test_var_name = f"{var}_{test}"
            test_name_list.append(test_var_name)
            ds[test_var_name] = ds[var].str[i].astype(int)

        ds = ds.drop(var)
    logger.info(f"ds size post coercion: {ds.nbytes}")
    logger.info(f"removing these as a test {test_name_list}")
    ds = ds.drop_vars(test_name_list)
    logger.info(ds.variables)
    return ds