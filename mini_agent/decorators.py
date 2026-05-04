def my_decorator(fn):
    def inner(prompt):
        print("before call")

        result = fn(prompt)   # original function

        print("after call")

        return result

    return inner
def retry(max_retries=3):
    def decorator(fn):
        def inner(prompt):
            for i in range(max_retries):
                try:
                    print(f"attempt {i+1}")
                    return fn(prompt)
                except Exception:
                    pass
            return "failed"
        return inner
    return decorator


@retry(3)
def call_llm(prompt):
    return f"LLM: {prompt}"


def wrapped_call(prompt):
    print("before call")          # BEFORE behavior

    result = call_llm(prompt)     # ORIGINAL function

    print("after call")           # AFTER behavior
    return result



def retry(fn):
    def wrapper(prompt):
        for i in range(3):
            start = time.time()
            try:
                result = fn(prompt)
                end = time.time()
                print("attempt time:", end - start)
                return result
            except:
                pass
        return "failed"
    return wrapper


def logger(fn):
    def wrapper(prompt):
        print(f"[log] input: {prompt}")
        result = fn(prompt)
        print(f"[log] output: {result}")
        return result
    return wrapper


import time

def timer(fn):
    def wrapper(prompt):
        start = time.time()
        result = fn(prompt)
        end = time.time()
        print(f"[time] {end - start:.4f}s")
        return result
    return wrapper


def fallback(fn):
    def wrapper(prompt):
        try:
            return fn(prompt)
        except:
            return "fallback response"
    return wrapper

def build_pipeline(middlewares):
     def wrap(fn):
            for middleware in reversed(middlewares):
                fn = middleware(fn)
            return fn
     return wrap
pipeline = build_pipeline([logger,timer,fallback,retry])
wrapped_llm =pipeline(call_llm)
result = wrapped_llm("Hi")


import time

def retry_with_trace(fn, max_retries=3):
    def wrapper(prompt):
        attempts = []
        start_total = time.time()

        for i in range(max_retries):
            start = time.time()

            try:
                result = fn(prompt)

                end = time.time()
                attempts.append({
                    "try": i + 1,
                    "time": end - start,
                    "status": "success"
                })

                break

            except Exception as e:
                end = time.time()
                attempts.append({
                    "try": i + 1,
                    "time": end - start,
                    "status": "fail"
                })

                result = None

        total_time = time.time() - start_total

        return {
            "result": result,
            "attempts": attempts,
            "total_time": total_time
        }

    return wrapper


@retry_with_trace
def call_llm(prompt):
    return f"LLM: {prompt}"


print(call_llm("abcd"))