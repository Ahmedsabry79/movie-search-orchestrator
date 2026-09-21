from datetime import datetime

from sqlalchemy import (
    BigInteger, CheckConstraint, Column, DateTime, Float, ForeignKey, Index,
    Integer, MetaData, String, Table, Text, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

SCHEMA = "movie_catalog"


class Base(DeclarativeBase):
    metadata = MetaData(schema=SCHEMA)


def association(name, foreign_name, target, foreign_type=BigInteger):
    return Table(name, Base.metadata,
        Column("movie_id", BigInteger, ForeignKey(f"{SCHEMA}.movies.id"), primary_key=True),
        Column(foreign_name, foreign_type, ForeignKey(f"{SCHEMA}.{target}"), primary_key=True),
        Index(f"ix_{name}_{foreign_name}", foreign_name),
    )


movie_genres = association("movie_genres", "genre_id", "genres.id")
movie_keywords = association("movie_keywords", "keyword_id", "keywords.id")
movie_production_companies = association("movie_production_companies", "company_id", "production_companies.id")
movie_production_countries = association("movie_production_countries", "country_iso_3166_1", "production_countries.iso_3166_1", String(2))
movie_spoken_languages = association("movie_spoken_languages", "language_iso_639_1", "spoken_languages.iso_639_1", String(8))


class Movie(Base):
    __tablename__ = "movies"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(Text)
    original_title: Mapped[str | None] = mapped_column(Text)
    title_search: Mapped[str] = mapped_column(Text)
    original_title_search: Mapped[str | None] = mapped_column(Text)
    tagline_search: Mapped[str | None] = mapped_column(Text)
    overview: Mapped[str | None] = mapped_column(Text)
    tagline: Mapped[str | None] = mapped_column(Text)
    homepage: Mapped[str | None] = mapped_column(Text)
    original_language: Mapped[str | None] = mapped_column(String(8))
    status: Mapped[str | None] = mapped_column(Text)
    release_date: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    budget: Mapped[int | None] = mapped_column(BigInteger)
    revenue: Mapped[int | None] = mapped_column(BigInteger, index=True)
    vote_count: Mapped[int | None] = mapped_column(BigInteger)
    vote_average: Mapped[float | None] = mapped_column(Float, index=True)
    popularity: Mapped[float | None] = mapped_column(Float)
    runtime: Mapped[float | None] = mapped_column(Float)
    missing_fields: Mapped[list] = mapped_column(JSONB)
    genres: Mapped[list["Genre"]] = relationship(secondary=movie_genres, viewonly=True)
    keywords: Mapped[list["Keyword"]] = relationship(secondary=movie_keywords, viewonly=True)
    production_companies: Mapped[list["Company"]] = relationship(secondary=movie_production_companies, viewonly=True)
    production_countries: Mapped[list["Country"]] = relationship(secondary=movie_production_countries, viewonly=True)
    spoken_languages: Mapped[list["Language"]] = relationship(secondary=movie_spoken_languages, viewonly=True)
    cast: Mapped[list["CastCredit"]] = relationship(viewonly=True)
    crew: Mapped[list["CrewCredit"]] = relationship(viewonly=True)
    __table_args__ = (
        *[CheckConstraint(f"{name} > 0", name=f"ck_movies_positive_{name}") for name in ("budget", "revenue", "vote_count", "vote_average", "runtime")],
        CheckConstraint("popularity >= 0", name="ck_movies_popularity"),
        *[Index(f"ix_movies_{name}_trgm", name, postgresql_using="gin", postgresql_ops={name: "gin_trgm_ops"}) for name in ("title_search", "original_title_search", "tagline_search")],
    )


class NamedEntity:
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    name: Mapped[str | None] = mapped_column(Text)
    name_search: Mapped[str | None] = mapped_column(Text, index=True)


class Genre(NamedEntity, Base):
    __tablename__ = "genres"


class Keyword(NamedEntity, Base):
    __tablename__ = "keywords"


class Company(NamedEntity, Base):
    __tablename__ = "production_companies"


class Country(Base):
    __tablename__ = "production_countries"
    iso_3166_1: Mapped[str] = mapped_column(String(2), primary_key=True)
    name: Mapped[str | None] = mapped_column(Text)
    name_search: Mapped[str | None] = mapped_column(Text)


class Language(Base):
    __tablename__ = "spoken_languages"
    iso_639_1: Mapped[str] = mapped_column(String(8), primary_key=True)
    name: Mapped[str | None] = mapped_column(Text)
    name_search: Mapped[str | None] = mapped_column(Text)


class Person(NamedEntity, Base):
    __tablename__ = "people"
    gender: Mapped[int | None] = mapped_column(Integer)
    __table_args__ = (Index("ix_people_name_search_trgm", "name_search", postgresql_using="gin", postgresql_ops={"name_search": "gin_trgm_ops"}),)


class CastCredit(Base):
    __tablename__ = "movie_cast"
    credit_id: Mapped[str] = mapped_column(Text, primary_key=True)
    movie_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(f"{SCHEMA}.movies.id"), index=True)
    person_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(f"{SCHEMA}.people.id"), index=True)
    character: Mapped[str | None] = mapped_column(Text)
    cast_order: Mapped[int | None] = mapped_column(Integer)
    person: Mapped[Person] = relationship()


class CrewCredit(Base):
    __tablename__ = "movie_crew"
    credit_id: Mapped[str] = mapped_column(Text, primary_key=True)
    movie_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(f"{SCHEMA}.movies.id"), index=True)
    person_id: Mapped[int] = mapped_column(BigInteger, ForeignKey(f"{SCHEMA}.people.id"), index=True)
    department: Mapped[str | None] = mapped_column(Text)
    job: Mapped[str | None] = mapped_column(Text, index=True)
    person: Mapped[Person] = relationship()


class DatasetState(Base):
    __tablename__ = "dataset_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    version: Mapped[str] = mapped_column(Text)
    row_counts: Mapped[dict] = mapped_column(JSONB)
    source_hashes: Mapped[dict] = mapped_column(JSONB)
    seeded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
