import os

def conditional_set_memory_limit(value):
    def wrapper(func):
        if os.name == 'nt':
            return set_memory_limit(value)(func)
        return func
    return wrapper


def default_set_memory_limit(memory_limit_mb):
	import win32api
	import win32job
	hjob = win32job.CreateJobObject(None, "")
	info = win32job.QueryInformationJobObject(hjob, win32job.JobObjectExtendedLimitInformation)
	info['ProcessMemoryLimit'] = memory_limit_mb * 1024 * 1024 * 1024
	info['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY

	win32job.SetInformationJobObject(hjob, win32job.JobObjectExtendedLimitInformation, info)

	hproc = win32api.GetCurrentProcess()
	win32job.AssignProcessToJobObject(hjob, hproc)

def set_memory_limit(memory_limit_gb=2):
	def set_memory_limit_wrapper(func):
		def wrapper(*args, **kwargs):
			# set memory limit
			import win32api
			import win32job
			hjob = win32job.CreateJobObject(None, "")
			info = win32job.QueryInformationJobObject(hjob, win32job.JobObjectExtendedLimitInformation)
			info['ProcessMemoryLimit'] = memory_limit_gb * 1024 * 1024 * 1024
			info['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY

			win32job.SetInformationJobObject(hjob, win32job.JobObjectExtendedLimitInformation, info)

			hproc = win32api.GetCurrentProcess()
			win32job.AssignProcessToJobObject(hjob, hproc)

			# run script
			func(*args, **kwargs)

		return wrapper
	return set_memory_limit_wrapper


@set_memory_limit(5)
def main():
	data = []
	while True:
		data.append('a' * 10*1)


if __name__ == '__main__':
	main()
