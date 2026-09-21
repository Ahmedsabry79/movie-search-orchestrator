from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .models import CastCredit, CrewCredit, Movie


def movie_details(connection, movie_id):
    with Session(bind=connection) as session:
        movie = session.scalar(select(Movie).where(Movie.id == movie_id).options(
            selectinload(Movie.genres), selectinload(Movie.keywords),
            selectinload(Movie.production_companies), selectinload(Movie.production_countries),
            selectinload(Movie.spoken_languages), selectinload(Movie.cast).selectinload(CastCredit.person),
            selectinload(Movie.crew).selectinload(CrewCredit.person),
        ))
        if movie is None:
            return None
        result = {column.name: getattr(movie, column.name) for column in Movie.__table__.columns if not column.name.endswith("_search")}
        for name, key in [("genres", "id"), ("keywords", "id"), ("production_companies", "id"),
                          ("production_countries", "iso_3166_1"), ("spoken_languages", "iso_639_1")]:
            result[name] = [{key: getattr(entity, key), "name": entity.name} for entity in sorted(getattr(movie, name), key=lambda entity: (entity.name or "", getattr(entity, key)))]
        result["cast"] = [
            {"credit_id": credit.credit_id, "person_id": credit.person_id, "name": credit.person.name,
             "character": credit.character, "cast_order": credit.cast_order}
            for credit in sorted(movie.cast, key=lambda item: (item.cast_order is None, item.cast_order or 0, item.credit_id))
        ]
        result["crew"] = [
            {"credit_id": credit.credit_id, "person_id": credit.person_id, "name": credit.person.name,
             "department": credit.department, "job": credit.job}
            for credit in sorted(movie.crew, key=lambda item: (item.job or "", item.person.name or "", item.credit_id))
        ]
        return result
