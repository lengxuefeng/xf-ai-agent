-- 创建数据库 xf-ai-agent，指定字符集为 utf8mb4 及排序规则
CREATE DATABASE IF NOT EXISTS `xf-ai-agent`
CHARACTER SET utf8mb4
COLLATE utf8mb4_general_ci;

-- 使用该数据库
USE `xf-ai-agent`;

-- 系统模型服务配置表（系统预定义的模型服务）
CREATE TABLE `t_model_setting` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `service_name` varchar(100) NOT NULL COMMENT '服务名称，如OpenAI、Google Gemini等',
  `service_type` varchar(50) NOT NULL COMMENT '服务类型标识，如openai、gemini等，用于程序识别',
  `service_url` varchar(500) NOT NULL COMMENT 'API服务基础地址，如https://api.openai.com/v1',
  `api_key_template` varchar(100) DEFAULT NULL COMMENT 'API密钥格式模板，如sk-xxx、AIxxx等，用于提示用户',
  `icon` varchar(50) DEFAULT 'FiCpu' COMMENT '服务图标名称，使用Feather Icons图标库',
  `models` json NOT NULL COMMENT '支持的模型列表，JSON格式存储，如["gpt-4","gpt-3.5-turbo"]',
  `description` text COMMENT '服务描述信息，可选字段',
  `is_system_default` tinyint(1) DEFAULT '1' COMMENT '是否为系统默认配置，true为系统预设，false为自定义',
  `is_enabled` tinyint(1) DEFAULT '1' COMMENT '是否启用该服务，true为启用，false为禁用',
  `create_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间，自动记录',
  `update_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间，自动更新',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='系统模型服务配置表';

-- 用户信息表
CREATE TABLE `t_user_info` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `token` varchar(255) DEFAULT NULL,
  `nick_name` varchar(50) NOT NULL COMMENT '昵称',
  `user_name` varchar(50) NOT NULL COMMENT '用户名',
  `phone` varchar(255) NOT NULL COMMENT '手机号',
  `password` varchar(255) NOT NULL COMMENT '密码',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='用户信息表';

-- 用户模型配置表（用户个人的模型配置）
CREATE TABLE `t_user_model` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `model_setting_id` bigint NOT NULL COMMENT '关联的系统模型服务ID',
  `service_name` varchar(100) NOT NULL COMMENT '服务名称（冗余字段，便于查询）',
  `selected_model` varchar(100) NOT NULL COMMENT '用户选择的具体模型名称',
  `api_key` varchar(255) NOT NULL COMMENT '用户的API密钥',
  `api_url` varchar(500) DEFAULT NULL COMMENT '用户自定义的API地址（可选）',
  `custom_config` json DEFAULT NULL COMMENT '用户自定义配置，JSON格式存储',
  `is_active` tinyint(1) DEFAULT '0' COMMENT '是否为当前激活的配置',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='用户模型配置表';

-- 用户MCP设置表
CREATE TABLE `t_user_mcp` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `mcp_setting_json` varchar(255) NOT NULL COMMENT 'MCP配置JSON',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='用户MCP设置表';