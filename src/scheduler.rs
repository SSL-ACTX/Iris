//! Cooperative preemption / reduction-budget helper (boilerplate)

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A very small wrapper that counts how many times the inner future has been polled
/// and forces a yield when the configured budget is reached.
pub struct ReductionLimiter<F> {
    inner: F,
    budget: usize,
    polled: usize,
}

impl<F> ReductionLimiter<F>
where
    F: Future,
{
    /// Wrap a future with a reduction `budget`.
    pub fn new(inner: F, budget: usize) -> Self {
        // enforce a minimum budget of 1 to avoid degenerate behavior
        let budget = if budget == 0 { 1 } else { budget };
        Self { inner, budget, polled: 0 }
    }
}

impl<F> Future for ReductionLimiter<F>
where
    F: Future,
{
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Obtain a mutable reference to the pinned struct without moving `inner`.
        let this = unsafe { self.as_mut().get_unchecked_mut() };

        if this.polled >= this.budget {
            this.polled = 0;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }

        this.polled += 1;

        // Project `inner` as a pinned reference and poll it.
        let inner = unsafe { Pin::new_unchecked(&mut this.inner) };
        inner.poll(cx)
    }
}

// Implement `Unpin` for the wrapper when the inner future is `Unpin`.
impl<F: Unpin> Unpin for ReductionLimiter<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use futures::task::noop_waker_ref;

    // Basic compile-time test that the wrapper can be used in an async context.
    #[tokio::test]
    async fn limiter_allows_completion() {
        async fn fast() { /* no-op */ }
        let fut = ReductionLimiter::new(fast(), 10);
        fut.await;
    }

    // Unit test that the limiter forces a `Poll::Pending` when the budget is
    // reached and still allows the inner future to complete eventually.
    #[test]
    fn limiter_forces_pending_before_ready() {
        struct Inner { polls_left: usize }
        impl std::future::Future for Inner {
            type Output = &'static str;
            fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
                if self.polls_left == 0 {
                    Poll::Ready("done")
                } else {
                    self.polls_left -= 1;
                    Poll::Pending
                }
            }
        }

        // inner requires 2 Poll::Pending calls before it becomes Ready
        let inner = Inner { polls_left: 2 };
        let wrapped = ReductionLimiter::new(inner, 1); // budget = 1

        let waker = noop_waker_ref();
        let mut cx = Context::from_waker(waker);
        let mut pinned = Box::pin(wrapped);

        // 1st poll -> limiter allows inner to be polled (inner -> Pending)
        assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Pending);

        // 2nd poll -> limiter detects budget reached and returns Pending WITHOUT polling inner
        assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Pending);

        // keep polling until the inner completes — the limiter will interleave
        // Pending responses when budget is hit. Ensure the inner eventually
        // completes within a small number of polls.
        let mut completed = false;
        for _ in 0..10 {
            match pinned.as_mut().poll(&mut cx) {
                Poll::Ready(_) => { completed = true; break; }
                Poll::Pending => continue,
            }
        }
        assert!(completed, "inner future did not complete within expected polls");
    }
}

