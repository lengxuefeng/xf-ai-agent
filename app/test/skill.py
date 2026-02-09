import uuid
from typing import TypedDict, NotRequired
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain.messages import SystemMessage
from langchain_community.chat_models import ChatZhipuAI
from langgraph.checkpoint.memory import InMemorySaver
from typing import Callable
from langchain_openai import ChatOpenAI


# 定义技能结构
class Skill(TypedDict):
    """一个可以逐步向代理（Agent）披露的技能。"""
    name: str
    description: str
    content: str


# 定义包含模式和业务逻辑的技能
SKILLS: list[Skill] = [
    {
        "name": "sales_analytics",  # 销售分析
        "description": "用于分析销售数据的数据库模式和业务逻辑，包括客户、订单和收入。",
        "content": """# 销售分析模式 (Sales Analytics Schema)

## 表结构 (Tables)

### customers (客户表)
- customer_id (主键)
- name (姓名)
- email (邮箱)
- signup_date (注册日期)
- status (状态: active/inactive - 活跃/非活跃)
- customer_tier (客户等级: bronze/silver/gold/platinum - 铜/银/金/白金)

### orders (订单表)
- order_id (主键)
- customer_id (外键 -> customers)
- order_date (订单日期)
- status (状态: pending/completed/cancelled/refunded - 待处理/已完成/已取消/已退款)
- total_amount (总金额)
- sales_region (销售区域: north/south/east/west - 北/南/东/西)

### order_items (订单明细表)
- item_id (主键)
- order_id (外键 -> orders)
- product_id (产品ID)
- quantity (数量)
- unit_price (单价)
- discount_percent (折扣百分比)

## 业务逻辑 (Business Logic)

**活跃客户 (Active customers)**: status = 'active' 且 signup_date <= 当前日期 - 间隔 '90 days'

**收入计算 (Revenue calculation)**: 仅计算 status = 'completed' 的订单。使用 orders 表中的 total_amount，该金额已包含折扣。

**客户生命周期价值 (CLV)**: 一个客户所有已完成订单金额的总和。

**高价值订单 (High-value orders)**: total_amount > 1000 的订单。

## 查询示例 (Example Query)

-- 获取上一季度收入最高的前10名客户
SELECT
    c.customer_id,
    c.name,
    c.customer_tier,
    SUM(o.total_amount) as total_revenue
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
  AND o.order_date >= CURRENT_DATE - INTERVAL '3 months'
GROUP BY c.customer_id, c.name, c.customer_tier
ORDER BY total_revenue DESC
LIMIT 10;
""",
    },
    {
        "name": "inventory_management",  # 库存管理
        "description": "用于库存跟踪的数据库模式和业务逻辑，包括产品、仓库和库存水平。",
        "content": """# 库存管理模式 (Inventory Management Schema)

## 表结构 (Tables)

### products (产品表)
- product_id (主键)
- product_name (产品名称)
- sku (库存单位)
- category (类别)
- unit_cost (单位成本)
- reorder_point (再订货点：重新订货前的最低库存水平)
- discontinued (布尔值：是否停产)

### warehouses (仓库表)
- warehouse_id (主键)
- warehouse_name (仓库名称)
- location (位置)
- capacity (容量)

### inventory (库存表)
- inventory_id (主键)
- product_id (外键 -> products)
- warehouse_id (外键 -> warehouses)
- quantity_on_hand (现有数量)
- last_updated (最后更新时间)

### stock_movements (库存变动表)
- movement_id (主键)
- product_id (外键 -> products)
- warehouse_id (外键 -> warehouses)
- movement_type (变动类型: inbound/outbound/transfer/adjustment - 入库/出库/转移/调整)
- quantity (数量：入库为正，出库为负)
- movement_date (变动日期)
- reference_number (参考编号)

## 业务逻辑 (Business Logic)

**可用库存 (Available stock)**: inventory 表中 quantity_on_hand > 0 的记录。

**需补货产品 (Products needing reorder)**: 所有仓库的现有数量 (quantity_on_hand) 总和小于或等于产品再订货点 (reorder_point) 的产品。

**仅限在售产品 (Active products only)**: 排除 discontinued = true 的产品，除非专门分析停产项目。

**库存估值 (Stock valuation)**: 每个产品的 quantity_on_hand * unit_cost。

## 查询示例 (Example Query)

-- 查找所有仓库中低于再订货点的产品
SELECT
    p.product_id,
    p.product_name,
    p.reorder_point,
    SUM(i.quantity_on_hand) as total_stock,
    p.unit_cost,
    (p.reorder_point - SUM(i.quantity_on_hand)) as units_to_reorder
FROM products p
JOIN inventory i ON p.product_id = i.product_id
WHERE p.discontinued = false
GROUP BY p.product_id, p.product_name, p.reorder_point, p.unit_cost
HAVING SUM(i.quantity_on_hand) <= p.reorder_point
ORDER BY units_to_reorder DESC;
""",
    },
]


# 创建技能加载工具
@tool
def load_skill(skill_name: str) -> str:
    """将技能的完整内容加载到代理（Agent）的上下文中。

    当你需要处理特定类型请求的详细信息时使用此工具。
    这将为你提供该技能领域的全面说明、策略和准则。

    Args:
        skill_name: 要加载的技能名称 (例如: "sales_analytics", "inventory_management")
    """
    # 查找并返回请求的技能

    for skill in SKILLS:
        if skill["name"] == skill_name:
            return f"已加载技能: {skill_name}\n\n{skill['content']}"

    # 未找到技能
    available = ", ".join(s["name"] for s in SKILLS)
    return f"未找到技能 '{skill_name}'。可用技能: {available}"


# 创建技能中间件
class SkillMiddleware(AgentMiddleware):
    """将技能描述注入系统提示词的中间件。"""

    # 将 load_skill 工具注册为类变量
    tools = [load_skill]

    def __init__(self):
        """初始化并从 SKILLS 列表生成技能提示词。"""
        # 从 SKILLS 列表构建技能提示词
        skills_list = []
        for skill in SKILLS:
            skills_list.append(
                f"- **{skill['name']}**: {skill['description']}"
            )
        self.skills_prompt = "\n".join(skills_list)

    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """同步：将技能描述注入到系统提示词中。"""
        # 构建技能补充内容
        skills_addendum = (
            f"\n\n## 可用技能 (Available Skills)\n\n{self.skills_prompt}\n\n"
            "当你需要处理特定类型请求的详细信息时，请使用 load_skill 工具。"
        )

        # 追加到系统消息内容块中
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)


# 初始化聊天模型 (请替换为你自己的模型配置)
# 示例: from langchain_anthropic import ChatAnthropic
# model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# 这里使用 GPT-4
# model = ChatOpenAI(model="gpt-4")
# model = ChatOpenAI(
model = ChatZhipuAI(
    model="GLM-4.6",
    temperature=0.5,
    api_key="372825e177b84308934ad6564cab60f7.1yrv1Sv8mL7YhHZA",
    # api_base="https://open.bigmodel.cn/api/paas/v4",
    # api_base="https://open.bigmodel.cn/api/coding/paas/v4",

)

# 创建支持技能的代理
agent = create_agent(
    model,
    system_prompt=(
        "你是一个 SQL 查询助手，帮助用户编写针对业务数据库的查询语句。"
    ),
    middleware=[SkillMiddleware()],
    checkpointer=InMemorySaver(),
)

# 使用示例
if __name__ == "__main__":
    # 配置此对话线程
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 请求编写 SQL 查询
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        # "写一个 SQL 查询，找出上个月订单金额超过 1000 美元的所有客户"
                        "写一个 SQL 查询，找出可用库存"
                    ),
                }
            ]
        },
        config
    )

    print(result["messages"])
    # 打印对话内容
    for message in result["messages"]:
        if hasattr(message, 'pretty_print'):
            message.pretty_print()
        else:
            print(f"{message.type}: {message.content}")
