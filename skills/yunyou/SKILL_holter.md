---
name: yunyou_user_holter
description: 用于分析用户holter数据的数据库模式和业务逻辑，包括用户、holter和报告信息。
---

##  表结构

###  t_holter_use_record（holter 记录表）
- id （主键）
- user_id （用户 ID）
- check_num （医院的 checkNum，大良卿接口返回）
- org_code （医院的 orgCode）
- apply_num （申请单号，大良卿接口返回）
- patient_id （患者 id，大良卿接口返回）
- hospital_id （医院 id，大良卿医院机构配置表）
- device_sn （设备 SN）
- begin_date_time （开始时间，非空）
- end_date_time （结束时间，非空）
- data_begin_time （报告数据查询开始时间）
- data_end_time （报告数据结束时间）
- use_day （使用日期，非空）
- is_uploaded （数据是否传完：0 = 否 / 1 = 是 /-1 = 没有数据，默认 0）
- is_data_download （是否数据已经下载完成（PC 审核平台需要下载数据），默认 0）
- hrv_cal_prog （已计算进度，默认 0）
- ecg_cal_prog （ecg 计算进度，默认 0）
- voltage （电量）
- report_status （报告审核状态：-1 = 无数据 / 0 = 待审核 / 1 = 审核中 / 2 = 人工审核完成 / 3 = 自动审核完成，非空，默认 0）
- record_status （记录状态：12=V2 审核 / 13=V3 审核）
- check_type （审核类型：1 = 人工审核 / 2 = 自动审核，非空，默认 1）
- pay_status （报告付款状态：0 = 待付款 / 1 = 已付款，非空，默认 0）
- app_type （app 类型：0=ECG 检测 app/1 = 小程序 / 2 = 云柚 app/10 = 第三方，非空，默认 0）
- det_status （检测状态 （目前只有小程序使用)：0 = 未完成 / 1 = 已完成，默认 0）
- auto_report_status （自动生成报告状态：0 = 未生成（未发送算法）/1 = 生成中（已发送算法）/2 = 已生成（算法已计算完成），默认 0）
- holter_type （holter 类型：0=24 小时 / 1=2 小时 / 2=24 小时（夜间）/3=48 小时，非空，默认 0）
- valid_min （有效数据分钟（如果大于此数字则提供给医生审核），非空，默认 0）
- upload_min （已上传文件个数，非空，默认 0）
- operator_id （操作人，非空，默认 0）
- file_generate_begin_time （文件生成开始时间）
- upload_time （数据上传结束时间）
- calc_finished_time （计算完成时间）
- template_time （模板计算完成时间）
- audit_time （报告审核完成时间）
- report_success （报告是否提交成功：0 = 否 / 1 = 是，默认 0）
- choose_report_types （下发时选择的报告类型，多个用，隔开：1 = 心理 / 2 = 精神 / 3 = 心脏 / 4 = 睡眠）
- template_status （是否计算完汤总的模板：0 = 否 / 1 = 是，非空，默认 0）
algorithm_status （算法计算返回状态：-4=Out of Memory/-3=（算法) 程序异常 /-2=ECGData 原始文件不存在或格式错误 /-1 = 检查软件授权 （License) 信息出错 / 0=ECGData 原始文件为空 / 1 = 处理成功 /）2=less - effective data/data error)
- result_storage （结果是否入库：0 = 否 / 1 = 是，默认 0）
- download_url （结果文件下载地址，非空，默认空字符串）
- admin_id （体检的 app 登录管理员 id）
- admin_company_id （体检 app 登录管理员的机构 ID）
- abn_number （异常次数）
- beat_number （总心博数）
- click_end_status （是否后台点击结束：0 = 否 / 1 = 是，默认 0）
- quality_info （holter 数据质量信息：frameLoss （丢帧数)，fallOff （脱落数)，totalLength （总长度)，默认空字符串）
- rr_scatter_url （散点图信息，JSON 格式：url = 图片地址 /num = 未审核个数 /hasCheckNum = 已审核个数 /btFlag 类型 = 0 = 正常 （N)/1 = 室上性 （S)/2 = 室性 （V)/26 = 房颤 （f)/-1 = 伪差 （-1)）
- report_summary （报告结论，默认空字符串，支持全文索引模糊查询）
- check_in_body （同济医院导验单 ORC 识别的信息，JSON 格式）
- other_info （其他信息，JSON 格式）
- extend_info （扩展字段信息，JSON 格式）
- send_algo_status （发送算法状态：0 = 否 / 1 = 是，默认 0）
- send_algo_type （发送算法分析类型）
- send_algo_time （发送算法时间）
- del_flag （删除标记：1 = 删除 / 0 = 未删除，非空，默认 0）
- remark （备注）
- v3_archived （v3 版本是否压缩成功：1 = 是 / 0 = 否，默认 0）
- version_id （版本号，v3 审核的时候可能有多个版本号，多个用，隔开）
- batch_number （批次号）
- add_time （创建时间，默认当前时间戳）
- update_time （最后修改时间，默认当前时间戳）



###  t_user（用户表）
- id （主键）
- sex （性别：1 = 男 / 2 = 女，非空，默认 0）
- user_status （用户状态，非空，默认 0）
- user_name （用户登录 userName，非空，默认空字符串，唯一索引）
- password （密码，非空，默认空字符串）
- height （身高，非空，默认 0）
- weight （体重，非空，默认 0）
- card_no （身份证，非空，默认空字符串）
- nick_name （昵称，默认空字符串）
- phone （手机号，非空，默认空字符串）
- email （邮箱，默认空字符串）
- head_img_url （头像 url，非空，默认空字符串）
- disease_history （病史信息（精神报告使用））
- info （备注，非空，默认空字符串）
- birth_date （出生日期）
- source （来源：0 = 云柚 / 1 = 灵犀 / 2 = 洋泽 / 3 = 体检，默认 0）
- add_time （创建时间，非空，默认当前时间戳）
- update_time （最后修改时间，非空，默认当前时间戳）


## 业务逻辑 

**2小时类型**: holter_type = 1 
  
**已审核**: report_status in (2,3)
  
**已上传完成**: is_uploaded = 1
  
**当天已审核**: report_status in (2,3) and use_day = "当天日期(yyyy-MM-dd格式)"

**算法计算完成**: is_uploaded =1 and  ecg_cal_prog>=100 and hrv_cal_prog>=100，如果是2小时的还需要 and holter_type = 1 反之为 and holter_type != 1

**用户年龄**: 使用出生日期计算年龄 TIMESTAMPDIFF(year,birth_date, now()) age


## 查询示例

```mysql
# 查询 holter 2小时已生成报告的条数
SELECT COUNT(1) FROM t_holter_use_record WHERE holter_type =1  and report_status in (2,3) ;

# 查询 当天做holter的用户和holter类型，只输出[holterId、用户昵称、上传完成状态、报告状态、审核版本、算法状态、模版状态]
SELECT t.id,tu.nick_name,t.is_uploaded,t.is_uploaded,t.report_status,t.record_status,t.algorithm_status,t.template_status 
FROM t_holter_use_record AS t
LEFT JOIN t_user AS tu
ON t.user_id = tu.id
WHERE t.use_day = "2025-03-03"

# 查询 用户年龄大于30岁的用户信息
SELECT 
  nick_name,
  TIMESTAMPDIFF(YEAR, birth_date, NOW()) AS age
FROM 
  t_user
WHERE 
  birth_date < DATE_SUB(NOW(), INTERVAL 30 YEAR);

```
