from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

class SkillEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill: str = Field(
        min_length=1,
        description="Name of the extracted skill"
    )
    evidence: str = Field(
        min_length=1,
        description="Exact supporting sentence from resume"
    )

class ResumeExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_name: Optional[str] = Field(
        default=None,
        description="Candidate's full name"
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Technical and professional skills"
    )
    years_of_experience: Optional[int] = Field(
        default=None,
        ge=0,
        le=50,
        description="Total explicitly supported years of experience"
    )
    education: list[str] = Field(
        default_factory=list,
        description="Candidate's educational qualifications"
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Strengths supported by resume evidence"
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Identifiable gaps supported by resume evidence"
    )
    skill_evidence: list[SkillEvidence] = Field(
        default_factory=list,
        description="Evidence supporting extracted skills"
    )

    @field_validator("skills")
    @classmethod
    def validate_skills(cls, skills):
        cleaned_skills = []
        for skill in skills:
            skill = skill.strip()
            if skill and skill.lower() not in [
                item.lower() for item in cleaned_skills
            ]:
                cleaned_skills.append(skill)
        return cleaned_skills

    @field_validator("education", "strengths", "gaps")
    @classmethod
    def clean_string_lists(cls, values):
        return [
            value.strip()
            for value in values
            if value.strip()
        ]

    @field_validator("candidate_name")
    @classmethod
    def clean_candidate_name(cls, value):
        if value is None:
            return None
        value = value.strip()
        return value if value else None

    @field_validator("skill_evidence")
    @classmethod
    def validate_skill_evidence(cls, evidence):
        for item in evidence:
            item.skill = item.skill.strip()
            item.evidence = item.evidence.strip()
        return evidence
