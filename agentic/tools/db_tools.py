"""
Database Tools for CultPass External System

This module provides tool functions for querying and modifying the CultPass
external database, including user lookups, subscription management,
experience searches, and reservation operations.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from langchain_core.tools import tool
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker

from data.models.cultpass import User, Subscription, Experience, Reservation


# Database connection
ENGINE = create_engine("sqlite:///data/external/cultpass.db")
Session = sessionmaker(bind=ENGINE)


def _get_session():
    """Get a new database session."""
    return Session()


@tool
def user_lookup_tool(email: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Look up user information from CultPass database.
    
    Args:
        email: User's email address
        user_id: User's unique ID
        
    Returns:
        User profile with subscription information or error message
        
    Example:
        user_lookup_tool(email="john@example.com")
        user_lookup_tool(user_id="user_123")
    """
    session = _get_session()
    
    try:
        # Query user
        if email:
            user = session.query(User).filter(User.email == email).first()
        elif user_id:
            user = session.query(User).filter(User.user_id == user_id).first()
        else:
            return {
                "error": "Either email or user_id must be provided",
                "success": False
            }
        
        if not user:
            return {
                "error": "User not found",
                "suggestion": "Check email spelling or user ID",
                "success": False
            }
        
        # Get subscription info
        subscription = session.query(Subscription).filter(
            Subscription.user_id == user.user_id
        ).first()
        
        result = {
            "success": True,
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
            "is_blocked": user.is_blocked,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "subscription": None
        }
        
        if subscription:
            result["subscription"] = {
                "subscription_id": subscription.subscription_id,
                "status": subscription.status,
                "tier": subscription.tier,
                "monthly_quota": subscription.monthly_quota,
                "started_at": subscription.started_at.isoformat() if subscription.started_at else None,
                "ended_at": subscription.ended_at.isoformat() if subscription.ended_at else None
            }
        
        return result
    
    except Exception as e:
        return {
            "error": f"Database error: {str(e)}",
            "success": False
        }
    finally:
        session.close()


@tool
def subscription_check_tool(user_id: str) -> Dict[str, Any]:
    """
    Check subscription status and quota for a user.
    
    Args:
        user_id: User's unique ID
        
    Returns:
        Subscription details with quota information
        
    Example:
        subscription_check_tool(user_id="user_123")
    """
    session = _get_session()
    
    try:
        subscription = session.query(Subscription).filter(
            Subscription.user_id == user_id
        ).first()
        
        if not subscription:
            return {
                "error": "Subscription not found for user",
                "success": False
            }
        
        # Count active reservations for quota calculation
        active_reservations = session.query(Reservation).filter(
            and_(
                Reservation.user_id == user_id,
                Reservation.status.in_(["confirmed", "pending"])
            )
        ).count()
        
        return {
            "success": True,
            "subscription_id": subscription.subscription_id,
            "user_id": subscription.user_id,
            "status": subscription.status,
            "tier": subscription.tier,
            "monthly_quota": subscription.monthly_quota,
            "quota_used": active_reservations,
            "quota_remaining": subscription.monthly_quota - active_reservations,
            "started_at": subscription.started_at.isoformat() if subscription.started_at else None,
            "ended_at": subscription.ended_at.isoformat() if subscription.ended_at else None
        }
    
    except Exception as e:
        return {
            "error": f"Database error: {str(e)}",
            "success": False
        }
    finally:
        session.close()


@tool
def experience_search_tool(
    location: Optional[str] = None,
    category: Optional[str] = None,
    date: Optional[str] = None,
    tier: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for available experiences/events.
    
    Args:
        location: Location/city to filter by
        category: Category to filter by (e.g., "yoga", "fitness")
        date: Date to filter by (YYYY-MM-DD format)
        tier: Tier restriction ("basic" or "premium")
        
    Returns:
        List of matching experiences
        
    Example:
        experience_search_tool(location="Bangalore", category="yoga")
    """
    session = _get_session()
    
    try:
        query = session.query(Experience)
        
        # Apply filters
        if location:
            query = query.filter(Experience.location.ilike(f"%{location}%"))
        
        if tier:
            if tier.lower() == "basic":
                query = query.filter(Experience.is_premium == False)
            elif tier.lower() == "premium":
                # Premium tier can access all
                pass
        
        # Filter by date if provided
        if date:
            try:
                target_date = datetime.fromisoformat(date)
                query = query.filter(Experience.when >= target_date)
            except ValueError:
                return {
                    "error": "Invalid date format. Use YYYY-MM-DD",
                    "success": False
                }
        
        # Only show experiences with available slots
        query = query.filter(Experience.slots_available > 0)
        
        experiences = query.all()
        
        results = []
        for exp in experiences:
            # Filter by category in title/description if provided
            if category and category.lower() not in exp.title.lower() and category.lower() not in exp.description.lower():
                continue
            
            results.append({
                "experience_id": exp.experience_id,
                "title": exp.title,
                "description": exp.description,
                "location": exp.location,
                "when": exp.when.isoformat() if exp.when else None,
                "slots_available": exp.slots_available,
                "is_premium": exp.is_premium
            })
        
        return {
            "success": True,
            "count": len(results),
            "experiences": results
        }
    
    except Exception as e:
        return {
            "error": f"Database error: {str(e)}",
            "success": False
        }
    finally:
        session.close()


@tool
def reservation_list_tool(user_id: str, status: Optional[str] = None) -> Dict[str, Any]:
    """
    List user's reservations.
    
    Args:
        user_id: User's unique ID
        status: Filter by status (optional): "confirmed", "pending", "cancelled"
        
    Returns:
        List of reservations
        
    Example:
        reservation_list_tool(user_id="user_123", status="confirmed")
    """
    session = _get_session()
    
    try:
        query = session.query(Reservation).filter(Reservation.user_id == user_id)
        
        if status:
            query = query.filter(Reservation.status == status)
        
        reservations = query.all()
        
        results = []
        for res in reservations:
            # Get experience details
            experience = session.query(Experience).filter(
                Experience.experience_id == res.experience_id
            ).first()
            
            result_item = {
                "reservation_id": res.reservation_id,
                "user_id": res.user_id,
                "experience_id": res.experience_id,
                "status": res.status,
                "created_at": res.created_at.isoformat() if res.created_at else None,
                "updated_at": res.updated_at.isoformat() if res.updated_at else None
            }
            
            if experience:
                result_item["experience"] = {
                    "title": experience.title,
                    "location": experience.location,
                    "when": experience.when.isoformat() if experience.when else None
                }
            
            results.append(result_item)
        
        return {
            "success": True,
            "count": len(results),
            "reservations": results
        }
    
    except Exception as e:
        return {
            "error": f"Database error: {str(e)}",
            "success": False
        }
    finally:
        session.close()


@tool
def reservation_create_tool(
    user_id: str,
    experience_id: str,
    slot_time: str
) -> Dict[str, Any]:
    """
    Create a new reservation for user.
    
    Args:
        user_id: User's unique ID
        experience_id: Experience to reserve
        slot_time: Time slot in ISO format
        
    Returns:
        Confirmation with reservation details
        
    Example:
        reservation_create_tool(
            user_id="user_123",
            experience_id="exp_456",
            slot_time="2024-01-20T18:00:00"
        )
    """
    session = _get_session()
    
    try:
        # Check user exists and is not blocked
        user = session.query(User).filter(User.user_id == user_id).first()
        if not user:
            return {
                "error": "User not found",
                "success": False
            }
        
        if user.is_blocked:
            return {
                "error": "User account is blocked",
                "success": False
            }
        
        # Check subscription and quota
        subscription = session.query(Subscription).filter(
            Subscription.user_id == user_id
        ).first()
        
        if not subscription or subscription.status != "active":
            return {
                "error": "No active subscription found",
                "success": False
            }
        
        # Count active reservations
        active_count = session.query(Reservation).filter(
            and_(
                Reservation.user_id == user_id,
                Reservation.status.in_(["confirmed", "pending"])
            )
        ).count()
        
        if active_count >= subscription.monthly_quota:
            return {
                "error": "Monthly quota exceeded",
                "quota_used": active_count,
                "quota_limit": subscription.monthly_quota,
                "success": False
            }
        
        # Check experience exists and has slots
        experience = session.query(Experience).filter(
            Experience.experience_id == experience_id
        ).first()
        
        if not experience:
            return {
                "error": "Experience not found",
                "success": False
            }
        
        if experience.slots_available <= 0:
            return {
                "error": "No slots available",
                "success": False
            }
        
        # Check tier restrictions
        if experience.is_premium and subscription.tier == "basic":
            return {
                "error": "Premium experience requires Premium tier subscription",
                "success": False
            }
        
        # Create reservation
        import uuid
        reservation_id = f"res_{uuid.uuid4().hex[:8]}"
        
        new_reservation = Reservation(
            reservation_id=reservation_id,
            user_id=user_id,
            experience_id=experience_id,
            status="confirmed"
        )
        
        # Update slots
        experience.slots_available -= 1
        
        session.add(new_reservation)
        session.commit()
        
        return {
            "success": True,
            "reservation_id": reservation_id,
            "user_id": user_id,
            "experience": {
                "experience_id": experience.experience_id,
                "title": experience.title,
                "location": experience.location,
                "when": experience.when.isoformat() if experience.when else None
            },
            "status": "confirmed",
            "message": f"Successfully created reservation for {experience.title}"
        }
    
    except Exception as e:
        session.rollback()
        return {
            "error": f"Failed to create reservation: {str(e)}",
            "success": False
        }
    finally:
        session.close()


@tool
def reservation_cancel_tool(reservation_id: str) -> Dict[str, Any]:
    """
    Cancel an existing reservation.
    
    Args:
        reservation_id: Reservation ID to cancel
        
    Returns:
        Confirmation with refunded slot
        
    Example:
        reservation_cancel_tool(reservation_id="res_abc123")
    """
    session = _get_session()
    
    try:
        # Find reservation
        reservation = session.query(Reservation).filter(
            Reservation.reservation_id == reservation_id
        ).first()
        
        if not reservation:
            return {
                "error": "Reservation not found",
                "success": False
            }
        
        if reservation.status == "cancelled":
            return {
                "error": "Reservation already cancelled",
                "success": False
            }
        
        # Get experience details
        experience = session.query(Experience).filter(
            Experience.experience_id == reservation.experience_id
        ).first()
        
        # Cancel reservation
        reservation.status = "cancelled"
        
        # Refund slot
        if experience:
            experience.slots_available += 1
        
        session.commit()
        
        return {
            "success": True,
            "reservation_id": reservation_id,
            "status": "cancelled",
            "message": "Reservation cancelled successfully",
            "slot_refunded": True
        }
    
    except Exception as e:
        session.rollback()
        return {
            "error": f"Failed to cancel reservation: {str(e)}",
            "success": False
        }
    finally:
        session.close()


@tool
def refund_processing_tool(
    reservation_id: str,
    reason: str,
    approved_by: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process refund for a reservation (RESTRICTED).
    
    Args:
        reservation_id: Reservation ID to refund
        reason: Reason for refund
        approved_by: Approver name (required for premium tier)
        
    Returns:
        Success confirmation or approval_required flag
        
    Example:
        refund_processing_tool(
            reservation_id="res_abc123",
            reason="user_request",
            approved_by="agent_john"
        )
    """
    session = _get_session()
    
    try:
        # Find reservation
        reservation = session.query(Reservation).filter(
            Reservation.reservation_id == reservation_id
        ).first()
        
        if not reservation:
            return {
                "error": "Reservation not found",
                "success": False
            }
        
        # Get user's subscription
        subscription = session.query(Subscription).filter(
            Subscription.user_id == reservation.user_id
        ).first()
        
        # Check if approval is required
        if subscription and subscription.tier == "premium":
            if not approved_by:
                return {
                    "success": False,
                    "approval_required": True,
                    "message": "Premium tier refunds require human approval",
                    "tier": "premium"
                }
        
        # Process refund (cancel reservation)
        reservation.status = "cancelled"
        
        # Refund slot
        experience = session.query(Experience).filter(
            Experience.experience_id == reservation.experience_id
        ).first()
        
        if experience:
            experience.slots_available += 1
        
        session.commit()
        
        return {
            "success": True,
            "reservation_id": reservation_id,
            "status": "refunded",
            "reason": reason,
            "approved_by": approved_by,
            "message": "Refund processed successfully"
        }
    
    except Exception as e:
        session.rollback()
        return {
            "error": f"Failed to process refund: {str(e)}",
            "success": False
        }
    finally:
        session.close()
