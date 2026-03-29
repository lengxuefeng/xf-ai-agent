importsubprocessimportos#定义Java代码字符串java_code="""publicclassMain{publicstaticvoidmain(String[]args){System.out.println("HellofromJava!");intresult=add(10,20);System.out.println("10+20="+result);}publicstaticintadd(inta,intb){returna+b;}}"""filename="Main.java"#将Java代码写入文件withopen(filename,"w")asf:f.write(java_code)try:#编译Java代码print("---正在编译---")compile_process=subprocess.run(["javac",filename],capture_output=True,text=True)ifcompile_process.returncode!=0:print("编译错误:")print(compile_process.stderr)else:print("编译成功.\n")#运行Java代码print("---正在运行---")run_process=subprocess.run(["java","Main"],capture_output=True,text=True)print(run_process.stdout)ifrun_process.stderr:print("运行时错误:")print(run_process.stderr)exceptFileNotFoundError:print("错误:系统中未找到'javac'。请确保已安装JDK并配置了环境变量。")finally:#清理生成的文件ifos.path.exists(filename):os.remove(filename)ifos.path.exists("Main.class"):os.remove("Main.class")已按要求生成代码，如需运行，请明确说明要执行或测试这段代码。

```python
import subprocess
import os

# 定义 Java 代码字符串
java_code = """
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello from Java!");
        int result = add(10, 20);
        System.out.println("10 + 20 = " + result);
    }

    public static int add(int a, int b) {
        return a + b;
    }
}
"""

filename = "Main.java"

# 将 Java 代码写入文件
with open(filename, "w") as f:
    f.write(java_code)

try:
    # 编译 Java 代码
    print("--- 正在编译 ---")
    compile_process = subprocess.run(["javac", filename], capture_output=True, text=True)
    
    if compile_process.returncode != 0:
        print("编译错误:")
        print(compile_process.stderr)
    else:
        print("编译成功.\n")
        
        # 运行 Java 代码
        print("--- 正在运行 ---")
        run_process = subprocess.run(["java", "Main"], capture_output=True, text=True)
        print(run_process.stdout)
        
        if run_process.stderr:
            print("运行时错误:")
            print(run_process.stderr)

except FileNotFoundError:
    print("错误: 系统中未找到 'javac'。请确保已安装 JDK 并配置了环境变量。")

finally:
    # 清理生成的文件
    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists("Main.class"):
        os.remove("Main.class")
```