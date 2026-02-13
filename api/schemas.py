# api/schemas.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
import re

class EmployeeData(BaseModel):
    Age: int = Field(..., ge=18, le=70, description="Employee age")
    BusinessTravel: str = Field(..., pattern="^(Non-Travel|Travel_Rarely|Travel_Frequently)$")
    DailyRate: int = Field(..., ge=100, le=1500)
    Department: str = Field(..., pattern="^(Sales|Research & Development|Human Resources)$")
    DistanceFromHome: int = Field(..., ge=0, le=50)
    Education: int = Field(..., ge=1, le=5)
    EducationField: str
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    Gender: str = Field(..., pattern="^(Male|Female)$")
    HourlyRate: int = Field(..., ge=30, le=100)
    JobInvolvement: int = Field(..., ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5)
    JobRole: str
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MaritalStatus: str = Field(..., pattern="^(Single|Married|Divorced)$")
    MonthlyIncome: int = Field(..., ge=1000, le=20000)
    MonthlyRate: int = Field(..., ge=2000, le=27000)
    NumCompaniesWorked: int = Field(..., ge=0, le=9)
    OverTime: str = Field(..., pattern="^(Yes|No)$")
    PercentSalaryHike: int = Field(..., ge=11, le=25)
    PerformanceRating: int = Field(..., ge=3, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TotalWorkingYears: int = Field(..., ge=0, le=40)
    TrainingTimesLastYear: int = Field(..., ge=0, le=6)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0, le=40)
    YearsInCurrentRole: int = Field(..., ge=0, le=18)
    YearsSinceLastPromotion: int = Field(..., ge=0, le=15)
    YearsWithCurrManager: int = Field(..., ge=0, le=17)
    
    # Pydantic v2 uses field_validator (not @validator)
    @field_validator('EducationField')
    def validate_education_field(cls, v):
        valid_fields = ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 
                       'Human Resources', 'Other']
        if v not in valid_fields:
            raise ValueError(f'Must be one of {valid_fields}')
        return v
    
    @field_validator('JobRole')
    def validate_job_role(cls, v):
        valid_roles = ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                      'Manufacturing Director', 'Healthcare Representative', 'Manager',
                      'Sales Representative', 'Research Director', 'Human Resources']
        if v not in valid_roles:
            raise ValueError(f'Must be one of {valid_roles}')
        return v

class PredictionResponse(BaseModel):
    attrition_risk: int
    probability: float
    risk_level: str
    confidence_score: float
    timestamp: datetime = datetime.now()

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
