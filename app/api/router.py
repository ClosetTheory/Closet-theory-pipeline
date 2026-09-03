"""API v1 Router aggregation."""

from fastapi import APIRouter
from app.api.v1.images import router as images_router
from app.api.v1.garments import router as garments_router
from app.api.v1.compatibility import router as compatibility_router

api_router = APIRouter()
api_router.include_router(images_router)
api_router.include_router(garments_router)
api_router.include_router(compatibility_router)
