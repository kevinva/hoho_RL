import asyncio
import time
from tracemalloc import start


def display(num):
    print(f'等待：{num}秒')
    time.sleep(num)
    print('hello')

async def displayv2(num):
    print(f'等待：{num}秒')
    await asyncio.sleep(num)
    print('hello')


def running1():
    async def test1():
        print('1')
        await test2()
        print('2')
    async def test2():
        print('3')
        print('4')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test1())

def func(url): 
    print(f'正在对{url}发起请求:') 
    print(f'请求{url}成功!')

async def func1(url):
    print(f'正在对{url}发起请求:') 
    print(f'请求{url}成功!')

async def do_some_work(i, n): #使用async关键字定义异步函数
    print('任务{}等待: {}秒'.format(i, n))
    await asyncio.sleep(n) #休眠一段时间
    return '任务{}在{}秒后返回结束运行'.format(i, n)


if __name__ == '__main__':
    # running1()

    # for i in range(5):
    #     # display(1)
    #     asyncio.run(displayv2(1))

    # func('www.baidu.com')

    # c = func1('www.baidu.com')
    # loop = asyncio.get_event_loop()
    # task = loop.create_task(c)
    # loop.run_until_complete(task)
    # print(task)

    start_time = time.time()
    tasks = [
        asyncio.ensure_future(do_some_work(1, 2)),
        asyncio.ensure_future(do_some_work(2, 1)),
        asyncio.ensure_future(do_some_work(3, 3))
    ]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))
    for task in tasks:
        print('任务执行结果：', task.result())
    print('运行时间：', time.time() - start_time)


