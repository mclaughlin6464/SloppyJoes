! -*- f90 -*-
! file lbfgs.f90
! Approximate Hessian inversion
! version 0.1
SUBROUTINE lbfgs(n, H_0, k, s, y, d)

!*****************************************************************
!
!   subroutine lbfgs
!   
!   Approximately apply the inverse Hessian to d.
!   m: dim of H_0 (mxm) and d (m)
!   d: Vector to mupltiply by. On exit, the result of the multiplication
!
!*****************************************************************
    IMPLICIT NONE
    
    INTEGER ::  k, n
    REAL (KIND=8) ::  H_0(n,n), s(n,k), y(n,k), d(n) 

    INTEGER :: i,j
    REAL (KIND=8) ::  a(k), b(k), rho(k) 

!    print *,'d old',d
!    print *, n,k
!    print *,'s',s
!    print *,'y',y

    do i=1,k-1
        !print *, y(:,i), s(:,i)
        !print *,1/DOT_PRODUCT(y(:,i), s(:,i))
        rho(i) = 1.0/DOT_PRODUCT(y(:,i), s(:,i))
    end do

    do i=k-1, 1,-1
        a(i) =  rho(i)*(DOT_PRODUCT(s(:,i), d))
        !print *,i,a(i), rho(i)
        d = d - a(i)*y(:,i) ! might need to modify here?
        !print *,'d',d
    end do

    !print *,'d',d
    !print *,'H0', H_0

    d = MATMUL(H_0, d)
    !print *,'d',d

    do i=1,k-1
       b(i) = rho(i)*(DOT_PRODUCT(y(:,i), d)) 
       !print *,i, b(i), rho(i)
       d = d + (a(i)-b(i))*s(:,i) ! might also need a tweak here
       !print *,'d',d
    end do

!    print *,'d new',d
!    print *,' '

END SUBROUTINE lbfgs

SUBROUTINE update_storage(n, k, istep, s, y, d_x, d_jac)

    IMPLICIT NONE

    INTEGER :: n,k,istep

    REAL(KIND=8) :: s(n,k), y(n,k)
    REAL(KIND=8) :: d_x(n), d_jac(n)

    INTEGER :: i

!    print *,'s',s
!    print *,'y',y

    if (istep .LE. k) then
        !no shuffling needed
        s(:,istep) = d_x
        y(:,istep) = d_jac
    else
        do i=2,k 
            s(:,i-1) = s(:,i)
            y(:,i-1) = y(:,i)
        end do
    s(:,k) = d_x
    y(:,k) = d_jac
    end if

!    print *,'s',s
!    print *,'y',y
!    print *,' '

END SUBROUTINE update_storage 
