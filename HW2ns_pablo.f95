!     Last change:  YG Yun   10 Sep 2022

module parameters
implicit none
   REAL, PARAMETER  	:: b = 0.99, d = 0.025, a = 0.36
   REAL, PARAMETER  	:: klb = 0.01, inc = 0.025, kub = 45.0
   INTEGER, PARAMETER 	:: length_grid_k = (kub-klb)/inc + 1
   REAL , PARAMETER 	:: toler   = 1.e-4									! Numerical tolerance
	 REAL, PARAMETER 		:: Zg = 1.25, Zb = 0.2
	 REAL, PARAMETER    :: Pgg = 0.977, Pgb = 0.023, Pbg = 0.074, Pbb = 0.926
end module

! ============================================================================================
module global
USE parameters
implicit none
   REAL 		:: Kgrid(length_grid_k), value(length_grid_k, 2), g_k(length_grid_k, 2)
   REAL                 :: vtmp(length_grid_k,length_grid_k, 2), value_new(length_grid_k, 2)
end module

! ============================================================================================

PROGRAM  HW2NonStochastic
   REAL 			:: total, etime, dist
   REAL, DIMENSION(2)  		:: elapsed

   call solution

   total=etime(elapsed)


	PRINT*,'--------------------------------------------------'
	PRINT*,'total time elpased =',total
	PRINT*,'--------------------------------------------------'


END PROGRAM HW2NonStochastic

! ============================================================================================
subroutine solution
USE parameters
USE global

   IMPLICIT  NONE

   INTEGER :: iter, index_k, index_kp
   REAL :: diff, k, kp, c(2), diffs(2)

   INTEGER :: i = 1


   do while (i<=length_grid_k)   !do loop for assigning capital grid K
     Kgrid(i) = klb + (i-1)*inc
     !write(*,*) i, Kgrid(i)
     i = i + 1
   end do


   iter = 1
   diff = 1000.d0
   value(:,1) = 0.*Kgrid		!Initial Value guess
	 value(:,2) = 0.*Kgrid		!Initial Value guess

	do while (diff>= toler)

		do index_k = 1, length_grid_k				! Capital grid
			k = Kgrid(index_k)
      vtmp(index_k,:,:) = -1.0e-16

			do index_kp = 1, length_grid_k
				kp = Kgrid(index_kp)
				c(1) = Zg*k**a+(1.-d)*k-kp
				c(2) = Zb*k**a+(1.-d)*k-kp
				if (c(1)>0.) then
	                        	vtmp(index_k,index_kp,1) = log(c(1))+b*(Pgg*value(index_kp,1)+Pgb*value(index_kp,2))

	          		endif
				if (c(2)>0.) then
	                        	vtmp(index_k,index_kp,2) = log(c(2))+b*(Pbg*value(index_kp,1)+Pbb*value(index_kp,2))

				    		endif

			enddo

			value_new(index_k,1) = MAXVAL(vtmp(index_k,:,1))
			value_new(index_k,2) = MAXVAL(vtmp(index_k,:,2))
      g_k(index_k,1)  	   = Kgrid(MAXLOC(vtmp(index_k,:,1),1))
			g_k(index_k,2)  	   = Kgrid(MAXLOC(vtmp(index_k,:,2),1))

                enddo

    diffs(1) = maxval(abs(value_new(:,1)-value(:,1)))/ABS(value_new(length_grid_k,1))
		diffs(2) = maxval(abs(value_new(:,2)-value(:,2)))/ABS(value_new(length_grid_k,2))
		diff  = maxval(diffs)
		value = value_new


		print*, 'Iteration =',iter,'sup_norm =',diff
                iter = iter+1

	enddo

	print *, ' '
	print *, 'Successfully converged with sup_norm ', diff
    	!print *, g_k

    !CALL vcDrawCurve@(d, Kgrid, g_k, length_grid_k)


        open (UNIT=1,FILE='valuefun',STATUS='replace')
	do index_k = 1, length_grid_k
        	WRITE(UNIT=1,FMT=*) value(index_k,:)
        end do
	close (UNIT=1)

	open (UNIT=2,FILE='policyfun',STATUS='replace')
do index_k = 1, length_grid_k
		WRITE(UNIT=2,FMT=*) g_k(index_k,:)
	end do
close (UNIT=2)

end subroutine
