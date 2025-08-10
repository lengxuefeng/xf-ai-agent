-- 创建数据库 xf-ai-agent，指定字符集为 utf8mb4 及排序规则
CREATE DATABASE IF NOT EXISTS `xf-ai-agent`
CHARACTER SET utf8mb4
COLLATE utf8mb4_general_ci;

-- 使用该数据库
USE `xf-ai-agent`;

-- 模型配置表
CREATE TABLE `t_model_setting` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `model_name` varchar(50) NOT NULL COMMENT '模型名称',
  `model_type` varchar(50) NOT NULL COMMENT '模型类型',
  `model_url` varchar(255) NOT NULL COMMENT '模型地址',
  `model_params` varchar(255) NOT NULL COMMENT '模型参数',
  `model_desc` varchar(255) NOT NULL COMMENT '模型描述',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='模型配置表';

-- 用户信息表
CREATE TABLE `t_user_info` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `token` varchar(50) NOT NULL COMMENT 'token',
  `nick_name` varchar(50) NOT NULL COMMENT '昵称',
  `user_name` varchar(50) NOT NULL COMMENT '用户名',
   `phone` varchar(50) NOT NULL COMMENT '手机号',
  `password` varchar(50) NOT NULL COMMENT '密码',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户信息表';

-- 用户模型设置表
CREATE TABLE `t_user_model` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `model_setting_id` bigint(20) NOT NULL COMMENT '模型配置ID',
  `model_name` varchar(100) NOT NULL COMMENT '模型名称，多个逗号分隔',
  `api_key` varchar(255) NOT NULL COMMENT 'api密钥',
  `api_url` varchar(255) NOT NULL COMMENT 'api地址',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户模型设置表';

-- 用户MCP设置表
CREATE TABLE `t_user_mcp` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `mcp_setting_json` varchar(255) NOT NULL COMMENT 'MCP配置JSON',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户MCP设置表';