from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
from app.services.database_service import DatabaseService, User
from sqlalchemy.orm import Session
import os

router = APIRouter()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token creation
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class UserCreate(BaseModel):
    email: str
    password: str
    softwareBackground: Optional[str] = None
    hardwareBackground: Optional[str] = None
    experienceLevel: Optional[str] = None
    additionalInfo: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

@router.post("/auth/signup")
async def signup(user_data: UserCreate):
    db_service = DatabaseService()
    db = db_service.get_db()

    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Hash the password
        hashed_password = get_password_hash(user_data.password)

        # Create new user
        db_user = User(
            email=user_data.email,
            hashed_password=hashed_password,
            software_background=user_data.softwareBackground,
            hardware_background=user_data.hardwareBackground,
            experience_level=user_data.experienceLevel,
            additional_info=user_data.additionalInfo
        )

        db.add(db_user)
        db.commit()
        db.refresh(db_user)

        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": db_user.email}, expires_delta=access_token_expires
        )

        return {
            "user": {
                "id": db_user.id,
                "email": db_user.email,
                "softwareBackground": db_user.software_background,
                "hardwareBackground": db_user.hardware_background,
                "experienceLevel": db_user.experience_level
            },
            "token": access_token
        }
    finally:
        db.close()

@router.post("/auth/signin")
async def signin(user_credentials: UserLogin):
    db_service = DatabaseService()
    db = db_service.get_db()

    try:
        user = authenticate_user(db, user_credentials.email, user_credentials.password)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password"
            )

        if not user.is_active:
            raise HTTPException(status_code=400, detail="Inactive user")

        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )

        return {
            "user": {
                "id": user.id,
                "email": user.email,
                "softwareBackground": user.software_background,
                "hardwareBackground": user.hardware_background,
                "experienceLevel": user.experience_level
            },
            "token": access_token
        }
    finally:
        db.close()