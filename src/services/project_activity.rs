use std::time::Duration;

use sqlx::PgPool;
use uuid::Uuid;

/// Refresh sliding project expiry for meaningful user activity. The five
/// minute write window prevents audio segment requests from hammering Postgres.
pub async fn touch_project(
    pool: &PgPool,
    project_id: Uuid,
    ttl: Duration,
) -> Result<bool, sqlx::Error> {
    let seconds = i64::try_from(ttl.as_secs()).unwrap_or(i64::MAX);
    let updated = sqlx::query(
        "UPDATE projects SET last_activity_at=now(), expires_at=now()+($2 * interval '1 second'), updated_at=now() WHERE id=$1 AND deleting_at IS NULL AND expires_at>now() AND last_activity_at<=now()-interval '5 minutes'",
    )
    .bind(project_id)
    .bind(seconds)
    .execute(pool)
    .await?;
    if updated.rows_affected() > 0 {
        return Ok(true);
    }
    sqlx::query_scalar::<_, bool>(
        "SELECT EXISTS(SELECT 1 FROM projects WHERE id=$1 AND deleting_at IS NULL AND expires_at>now())",
    )
    .bind(project_id)
    .fetch_one(pool)
    .await
}
