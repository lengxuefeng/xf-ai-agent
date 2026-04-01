# -*- coding: utf-8 -*-
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from common.core.security import verify_token
from db import get_db
from models.schemas.response_model import ResponseModel
from models.schemas.user_skill_schemas import UserSkillCreate, UserSkillOut, UserSkillUpdate
from services.user_skill_service import user_skill_service

router = APIRouter(prefix="/skills", tags=["Agent Skills"])


@router.get("/", response_model=ResponseModel[List[UserSkillOut]], summary="获取当前用户的技能配置列表")
def get_user_skills(
    db: Session = Depends(get_db),
    user_id: int = Depends(verify_token),
):
    user_skills = user_skill_service.get_user_skills(db, user_id=user_id)
    return ResponseModel.success(data=user_skills)


@router.post("/", response_model=ResponseModel[UserSkillOut], summary="创建新的技能配置")
def create_user_skill(
    user_skill_req: UserSkillCreate,
    db: Session = Depends(get_db),
    user_id: int = Depends(verify_token),
):
    user_skill = user_skill_service.create_user_skill(db, user_skill=user_skill_req, user_id=user_id)
    return ResponseModel.success(data=user_skill, message="技能配置创建成功")


@router.put("/{skill_id}", response_model=ResponseModel[UserSkillOut], summary="更新技能配置")
def update_user_skill(
    skill_id: int,
    user_skill_req: UserSkillUpdate,
    db: Session = Depends(get_db),
    user_id: int = Depends(verify_token),
):
    try:
        updated_skill = user_skill_service.update_user_skill(db, skill_id=skill_id, user_skill=user_skill_req, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ResponseModel.success(data=updated_skill, message="技能配置更新成功")


@router.delete("/{skill_id}", response_model=ResponseModel[dict], summary="删除技能配置")
def delete_user_skill(
    skill_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(verify_token),
):
    try:
        user_skill_service.remove_user_skill(db, skill_id=skill_id, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ResponseModel.success(data={"message": "删除成功"})
