importjava.time.LocalDateTime;importjava.time.format.DateTimeFormatter;publicclassQueryTime{publicstaticvoidmain(String[]args){//获取当前时间LocalDateTimenow=LocalDateTime.now();//定义日期时间格式DateTimeFormatterformatter=DateTimeFormatter.ofPattern("yyyy-MM-ddHH:mm:ss");//格式化输出StringformattedDateTime=now.format(formatter);System.out.println("当前时间是:"+formattedDateTime);}}已按要求生成代码，如需运行，请明确说明要执行或测试这段代码。

```python
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class QueryTime {
    public static void main(String[] args) {
        // 获取当前时间
        LocalDateTime now = LocalDateTime.now();
        
        // 定义日期时间格式
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        
        // 格式化输出
        String formattedDateTime = now.format(formatter);
        
        System.out.println("当前时间是: " + formattedDateTime);
    }
}
```